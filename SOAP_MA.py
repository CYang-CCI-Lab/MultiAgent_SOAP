from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import re
from typing import Optional, Union, List, get_origin, get_args, Any, Dict, Literal, Callable
import inspect
import asyncio
import json
import logging
import pandas as pd
from pydantic import BaseModel, Field, create_model
import torch
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import chromadb
from chromadb.config import Settings
import math
from utils import safe_json_load, count_llama_tokens
from rag_retrieve import create_documents, hybrid_query
import os

LLAMA3_70B_MAX_TOKENS = 24000

logger = logging.getLogger(__name__)

selected_problems = [
    # 'congestive heart failure',
    # 'sepsis',
    'acute kidney injury',
]

class Response(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning leading to your choice.")
    choice: Literal["Yes", "No"] = Field(..., description="Your choice indicating whether the patient has the problem.")

class LLMAgent:
    def __init__(
        self, 
        system_prompt: str, 
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        client=AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy"),
        max_tokens: int = LLAMA3_70B_MAX_TOKENS,
        summarization_threshold: float = 0.8
    ):
        self.model_name = model_name
        self.client = client
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_tokens = max_tokens
        self.token_threshold = int(self.max_tokens * summarization_threshold)
        logger.info(f"[{self.__class__.__name__}] Initialized. Token limit: {self.max_tokens}, Summarization threshold: {self.token_threshold}")

    async def _summarize_history(self, inplace: bool = True, message: str = "") -> Union[bool, str]:
        logger.warning("[%s] Starting iterative summarization …", self.__class__.__name__)

        async def _summarize_once(text: str) -> Union[str, None]:
            summary_prompt = (
                "Summarize the following message concisely, "
                "preserving all key facts and reasoning steps. "
                "Do not exceed 1000 words.\n\n"
                "<<<MESSAGE_START>>>\n"
                f"{text}\n"
                "<<<MESSAGE_END>>>"
            )
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a summarization assistant."},
                        {"role": "user",   "content": summary_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1500,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error("Summarization call failed: %s", e)
                return None

        if inplace:  
            if len(self.messages) < 3:
                logger.warning("Not enough messages to summarise.")
                return False

            def longest_idx() -> int:
                return max(
                    (
                        (i, count_llama_tokens([m]))
                        for i, m in enumerate(self.messages)
                        if m["role"] != "system"
                    ),
                    key=lambda t: t[1],
                )[0]

            failures = 0
            while count_llama_tokens(self.messages) >= self.token_threshold:
                idx = longest_idx()
                summary = await _summarize_once(self.messages[idx]["content"])
                if summary is None:
                    return False

                # sanity‑check: summary must be shorter
                if count_llama_tokens([{"role": "assistant", "content": summary}]) >= \
                count_llama_tokens([self.messages[idx]]):
                    failures += 1
                    if failures > 3:
                        logger.error("Summarization failed repeatedly. Aborting.")
                        return False
                    continue

                self.messages[idx]["content"] = f"[Summary] {summary}"
                logger.info("Replaced longest message with summary (%d tokens total).",
                            count_llama_tokens(self.messages))
            return True

        # ── summarise an external `message` string ───────────────────────────────────────────
        if not message:
            logger.warning("No message provided for summarization.")
            return False
        
        summary = await _summarize_once(message)
        if summary is None:
            return False
        message = summary

        failures = 0
        while count_llama_tokens(message) >= self.token_threshold:
            summary = await _summarize_once(message)
            if summary is None:
                return False
            if count_llama_tokens(summary) >= count_llama_tokens(message):
                failures += 1
                if failures > 3:
                    logger.error("Summarization failed repeatedly. Aborting.")
                    return False
                continue
            message = summary     

        return message


    async def _llm_call_with_tools(self, params: dict, available_tools: dict) -> Any:
        # Check tokens before making the call
        if count_llama_tokens(params["messages"]) > self.token_threshold:
            await self._summarize_history()
            # Update params with potentially summarized messages
            params["messages"] = self.messages 
            # Add a check to ensure summarization didn't make it worse or fail
            if count_llama_tokens(params["messages"]) > self.max_tokens:
                 logger.error("Summarization failed to reduce tokens below limit. Aborting call.")
                 return "Error: Context limit exceeded even after summarization attempt."

        iter_count = 0
        try:
            response = await self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error(f"Error calling LLM with tools: {e}")
            return None
        
        message = response.choices[0].message
          
        while not hasattr(message, "tool_calls") or not message.tool_calls:
            logger.info("No tool calls detected in response.")
      
            self.messages.append({"role": "assistant", "content": message.content})
            logger.info(f"[{self.__class__.__name__}] memory token length after append: {count_llama_tokens(self.messages)}")

            # Provide a hint/warning
            self.messages.append({"role": "system", "content": "WARNING! Remember to call a tool."})
            logger.info(f"[{self.__class__.__name__}] memory token length after append: {count_llama_tokens(self.messages)}")

            iter_count += 1
            if iter_count > 3:
                logger.error("Exceeded maximum iterations for tool calls.")
                return message.content
            
            try:
                response = await self.client.chat.completions.create(**params)
            except Exception as e:
                logger.error(f"Error calling LLM with tools (iteration): {e}")
                return None

            message = response.choices[0].message

        logger.info("Tool calls detected in response.")
        self.messages.append({"role": "assistant", "tool_calls": message.tool_calls})
        logger.info(f"[{self.__class__.__name__}] memory token length after append: {count_llama_tokens(self.messages)}")  # <-- Logging memory size
        
        # Iterate over each detected tool call
        for call in message.tool_calls:
            logger.debug(f"Tool call: {call}")
            
            args = safe_json_load(call.function.arguments)
            if args is None:
                logger.error(f"Failed to parse tool call arguments: {call.function.arguments}")
                return

            # Ensure the tool is available
            if call.function.name not in available_tools:
                logger.error(f"Requested tool '{call.function.name}' not found in available_tools.")
                # Decide how to handle: skip or return an error. Here we skip.
                continue

            try:
                result = available_tools[call.function.name](**args)  # synchronous
            except Exception as e:
                logger.error(f"Error calling tool '{call.function.name}': {e}")
                return message.content
            
            self.messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": call.id,
                "name": call.function.name,
            })
            logger.info(f"[{self.__class__.__name__}] memory token length after append: {count_llama_tokens(self.messages)}")  # <-- Logging memory size
        
        try:
            response = await self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error(f"Error calling LLM after tool usage: {e}")
            return None

        # Potentially append this final message before returning? let the caller do it for now.
        return response.choices[0].message.content

    async def llm_call(
        self, 
        user_prompt: str, 
        temperature: float = 0.3,
        guided_: dict = None,
        tools_descript: List[dict] = None, 
        available_tools: dict = None
    ) -> Any:

        potential_new_length = count_llama_tokens(self.messages + [{"role": "user", "content": user_prompt}])
        if potential_new_length > self.token_threshold:
             summarized = await self._summarize_history()
             if not summarized and potential_new_length > self.max_tokens:
                  logger.error("History too long and summarization failed. Aborting call.")
                  raise ValueError("Context limit exceeded even after summarization attempt.")
        
        self.messages.append({"role": "user", "content": user_prompt})

        params = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": temperature,
        }
        if guided_:
            params["extra_body"] = guided_

        if tools_descript:
            params["tools"] = tools_descript
            assert available_tools is not None, "available_tools must be provided if tools_descript is used."
            return await self._llm_call_with_tools(params, available_tools)
        else:
            params["tool_choice"] = "none"
            try:
                response = await self.client.chat.completions.create(**params)
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error calling LLM in llm_call(): {e}")
                raise e
            
    def append_message(self, content: Any, role='assistant'): # inplace
        logger.debug(f"Appending message with role='{role}' to conversation.")
        self.messages.append({"role": role, "content": str(content)})
        # We won't trigger summarization here, but rely on checks before LLM calls.
        # If append causes it to exceed max_tokens, the next LLM call will fail/summarize.


class BaselineZS(LLMAgent):
    def __init__(self):
        super().__init__("You are a clinical reasoning assistant.")
        self.schema = None   # filled lazily

    async def analyze_note(self, note: str, problem: str):
        if self.schema is None:                   # build schema once
            self.schema = Response.model_json_schema()

        user_prompt = (
            "Read the patient note and decide whether the patient has the specified problem.\n"
            f"<<<PATIENT NOTE>>>\n{note}\n<<<END NOTE>>>\n\n"
            f"Question: Does this patient have {problem}?\n"
            "Please give your reasoning, then the choice ('Yes' or 'No')."
        )
        try:
            raw = await self.llm_call(
                user_prompt, 
                temperature=0.1, 
                guided_={"guided_json": self.schema}
            )
            parsed = safe_json_load(raw)
            return parsed
        except Exception as e:
            logger.error("[BaselineZS] analyze_note() failed: %s", e)
            return None

class Manager(LLMAgent):
    def __init__(
        self, 
        note: str, 
        hadm_id: str, 
        problem: str, 
        label: str, 
        n_specialists: Union[int, Literal["auto"]] = 'auto',
        n_generic_agents: int = 0, 
        consensus_threshold: float = 0.8,
        max_consensus_attempts=3, 
        max_assignment_attempts=2,
    ):
        system_prompt = (          
                "You are the manager of a multi‐agent diagnostic system. "
                "Your job is to coordinate sub‐agents to decide if the patient has the problem."
        )
        super().__init__(system_prompt)

        self.note = note
        self.hadm_id = hadm_id
        self.problem = problem
        self.label = label
        
        self.state = {
            "note": note, 
            "hadm_id": hadm_id, 
            "problem": problem, 
            "label": label,
            "cached_messages": [],
            "static agents": {},
            "final": {}
        }
        self.n_specialists = n_specialists
        self.n_generic_agents = n_generic_agents

        self.consensus_threshold = consensus_threshold
    
        self.max_consensus_attempts = max_consensus_attempts

        self.max_assignment_attempts = max_assignment_attempts
        self.assignment_attempts = 0  # panel

    
    async def _assign_specialists(self):
        self.assignment_attempts += 1
        logger.info(f"Starting assignment attempt #{self.assignment_attempts}.")
        self.state[f"panel_{self.assignment_attempts}"] = {}

        if self.assignment_attempts > 1:
            previous_panel_history = json.dumps(self.state[f"panel_{self.assignment_attempts - 1}"]["Collected Specialists"], indent=4)
            summary = await self._summarize_history(inplace=False, message=previous_panel_history)
            if isinstance(summary, str):
                message=(f"The previous panel of specialists couldn’t reach consensus. "
                                                            f"Summary so far: \n{summary}\n\n"
                                                            f"So we’re now assembling a new panel."
                                                            )
            else:
                logger.warning("Failed to summarize previous panel history.")
                previous_panel = self.state[f"panel_{self.assignment_attempts - 1}"]["Collected Specialists"].keys()
                message = (
                    f"The previous panel of specialists ({previous_panel}) couldn’t reach consensus. "
                    f"So we’re now assembling a new panel."
                )
            self.append_message(content=message, role='system')


        specialties_lst = []

        trial_n = 0
        while specialties_lst == [] and trial_n < 3:
            if self.n_specialists == "auto":
                user_prompt = (
                    "Below are the Subjective (S) and Objective (O) sections from a SOAP note:\n\n"
                    f"<SOAP>\n{self.note}\n</SOAP>\n\n"
                    f"We need to determine if this patient has {self.problem}.\n\n"
                    "Please provide a list of medical specialties that should be involved."
                )
                class Specialties(BaseModel):
                    specialties: List[str] = Field(..., description="List of medical specialties.")
                try:
                    response = await self.llm_call(user_prompt, guided_={"guided_json": Specialties.model_json_schema()})
                    self.append_message(content=response)
                    parsed = safe_json_load(response)
                    specialties_lst = parsed["specialties"]
                except Exception as e:
                    logger.error(f"Failed to parse 'specialties' from LLM response: {e}")
                self.n_specialists = len(specialties_lst)

            else:
                user_prompt = (
                    "Below are the Subjective (S) and Objective (O) sections from a SOAP note:\n\n"
                    f"<SOAP>\n{self.note}\n</SOAP>\n\n"
                    f"We need to determine if this patient has {self.problem}.\n\n"
                    f"Please provide a list of {self.n_specialists} medical specialties that should be involved."
                )
                class Specialty(BaseModel):
                    specialty_name: str = Field(..., description="Name of the medical specialty.")
                specialties_dict = {f"specialty_{i+1}": (Specialty, ...) for i in range(self.n_specialists)}
                Specialties_N = create_model("Specialties_N", **specialties_dict)
                try:
                    response = await self.llm_call(user_prompt, guided_={"guided_json": Specialties_N.model_json_schema()})
                    self.append_message(content=response)
                    parsed = safe_json_load(response)
                    for i in range(self.n_specialists):
                        # Potential KeyError if "specialty_i" or 'name' is missing
                        name = parsed[f"specialty_{i+1}"]["specialty_name"]
                        specialties_lst.append(name)
                except Exception as e:
                    logger.error(f"Failed to parse specialties from LLM response: {e}")
            
            trial_n += 1

        self.state[f"panel_{self.assignment_attempts}"]["Initially Identified Specialties"] = specialties_lst # when something goes wrong, this is empty
        self.state[f"panel_{self.assignment_attempts}"]["Collected Specialists"] = {}

        # Request a panel of specialists for each identified specialty
        user_prompt = (
            "Based on the list of specialties you provided: "
            f"{specialties_lst}\n\n"
            f"Please assemble a panel of {self.n_specialists} specialists. Assign each specialist to exactly one of the above specialties.\n"
            "For each specialist, specify their job title (not their personal name) and list the relevant expertise areas related to this case."
        )

        class Specialist(BaseModel):
            specialist: str = Field(..., description="Official job title.")
            expertise: List[str] = Field(..., description="Their main areas of expertise relevant to this case.")

        panel_dict = {f"specialist_{i+1}": (Specialist, ...) for i in range(self.n_specialists)}
        SpecialistPanel = create_model("SpecialistPanel", **panel_dict)

        try:
            panel_response = await self.llm_call(user_prompt, guided_={"guided_json": SpecialistPanel.model_json_schema()})
            self.append_message(content=panel_response)
            specialists_dict = safe_json_load(panel_response)

            for key, specialist in specialists_dict.items():
                role = specialist["specialist"]
                expertise = specialist["expertise"]
             
                self.state[f"panel_{self.assignment_attempts}"]["Collected Specialists"][role] = {
                    "expertise": expertise, 
                    "answer_history": {}
                }
                logger.info(f"Specialist {role} assigned with expertise: {expertise}")
                
        except Exception as e:
            logger.error(f"Failed to parse specialists from LLM response: {e}")
            return None
        
        return self.state
    
    def _check_consensus_specialists(self, panel_id: int, round_id: int) -> Optional[str]:
        logger.info(f"Checking for consensus among specialists in panel_{panel_id}, round_{round_id}.")
        choice_counts = {}
        majority_count = math.ceil(self.n_specialists * self.consensus_threshold)

        for role, answ_hist in self.state[f"panel_{panel_id}"]["Collected Specialists"].items():
            final_choice = answ_hist["answer_history"][f"round_{round_id}"]['choice']
            choice_counts[final_choice] = choice_counts.get(final_choice, 0) + 1

        for choice, count in choice_counts.items():
            if count >= majority_count:
                logger.info(f"Consensus found on choice '{choice}' with {count}/{self.n_specialists} specialists.")
                return choice
        logger.info("No consensus found.")
        return None

    async def _aggregate(self, chat_history):
        logger.info(f"Token count for the entire chat history to be aggregated: {count_llama_tokens(chat_history)}")

        user_prompt = (
            "No consensus has been reached among the sub-agents.\n"
            "You now have access to the entire conversation histories.\n"
            "Please analyze each agent's reasoning and final choice, then provide a single, definitive answer.\n"
            "Your answer should be the one best supported by their collective reasoning.\n\n"
            "Below is the complete conversation history for each agent:\n\n"
            f"{json.dumps(chat_history, indent=4)}\n\n"
            "After reviewing this, please provide:\n"
            "1) A concise summary of the reasoning behind your final decision\n"
            "2) A single recommended choice: 'Yes' or 'No' "
            f"(indicating whether the patient has {self.problem})."
        )

        class AggregatedResponse(BaseModel):
            final_reasoning: str = Field(..., description="Step-by-step reasoning leading to the final choice.")
            final_choice: Literal["Yes", "No"] = Field(..., description=f"Final choice indicating whether the patient has {self.problem}.")
        
        try:
            raw = await self.llm_call(user_prompt, temperature=0.1, guided_={"guided_json": AggregatedResponse.model_json_schema()})
            parsed = safe_json_load(raw)
            self.state["final"] = parsed
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse aggregator response: {e}")
            self.state["final"] = {"final_choice": "Error in aggregation.", "final_reasoning": "Unable to parse aggregator response."}
            return None
    
    async def run_specialists(self):
        while self.assignment_attempts < self.max_assignment_attempts:
            logger.info(f"Assignment attempt #{self.assignment_attempts + 1} started.")
            
            panel_state = await self._assign_specialists()  # increments self.assignment_attempts
            # 굳이 안반환해도 됨
            if panel_state is None:
                logger.error("Failed to assign specialists; retrying.")
                continue
          
            panel = []
            for role in self.state[f"panel_{self.assignment_attempts}"]["Collected Specialists"].keys():
                panel.append(
                    DynamicSpecialist(role, self.assignment_attempts, self.state)  # panel_id is the assignment attempt number
                )
                
            analyze_tasks = [asyncio.create_task(specialist.analyze_note(self.note, self.problem)) for specialist in panel]
            analyze_results = await asyncio.gather(*analyze_tasks)
            if any(r is None for r in analyze_results):
                logger.error("At least one specialist failed analysis; skipping this panel.")
                continue

            consensus_attempts = 1
            consensus_choice = self._check_consensus_specialists(self.assignment_attempts, consensus_attempts)
            if consensus_choice:
                self.state["final"] = {
                    "final_choice": consensus_choice, 
                    "final_reasoning": "Consensus reached"
                }
                return self.state
            
            # Debate loop
            while consensus_attempts < self.max_consensus_attempts:
                consensus_attempts += 1
                logger.info(f"Debate attempt #{consensus_attempts} started.")
                debate_tasks = [asyncio.create_task(specialist.debate()) for specialist in panel]
                debate_results = await asyncio.gather(*debate_tasks)
                if any(r is None for r in debate_results):
                    logger.error("At least one specialist failed during debate; skipping this round.")
                    continue
                
                consensus_choice = self._check_consensus_specialists(self.assignment_attempts, consensus_attempts)
                if consensus_choice:
                    self.state["final"] = {
                        "final_choice": consensus_choice, 
                        "final_reasoning": "Consensus reached"
                    }
                    return self.state
            logger.info("No consensus reached after maximum consensus attempts among the panel.")

        logger.info("No consensus reached after maximum assignment attempts. Proceeding to aggregation.")
        specialists_chat_history = self.state.copy()
        for field_to_remove in ["label", "hadm_id", "cached_messages", "static agents", "final"]:
            specialists_chat_history.pop(field_to_remove, None)
        aggregated_response = await self._aggregate(specialists_chat_history)
        if aggregated_response is None:
            logger.error("Aggregation failed; returning error.")
        return self.state
    
    async def run_generic_agents(self):
        panel = []
        for agent_id in [f"generic_agent_{i+1}" for i in range(self.n_generic_agents)]:
            panel.append(GenericAgent(agent_id, self.state))
            
        analyze_tasks = [asyncio.create_task(agent.analyse_note(self.note, self.problem)) for agent in panel]
        analyze_results = await asyncio.gather(*analyze_tasks)

        consensus_attempts = 1
        choice_counts = {}
        majority_count = math.ceil(self.n_generic_agents * self.consensus_threshold)
        for result in analyze_results:
            if result:
                choice_counts[result["choice"]] = choice_counts.get(result["choice"], 0) + 1

        for choice, count in choice_counts.items():
            if count >= majority_count:
                self.state["final"] = {
                    "final_choice": choice,
                    "final_reasoning": "Consensus reached"
                }
                return self.state
        
        # Debate loop
        while consensus_attempts < self.max_consensus_attempts:
            consensus_attempts += 1
            logger.info(f"Debate attempt #{consensus_attempts} started.")
            debate_tasks = [asyncio.create_task(agent.debate()) for agent in panel]
            debate_results = await asyncio.gather(*debate_tasks)
            if any(r is None for r in debate_results):
                logger.error("At least one agent failed during debate; skipping this round.")
                continue

            choice_counts = {}
            majority_count = math.ceil(self.n_generic_agents * self.consensus_threshold)
            for result in debate_results:
                choice_counts[result["choice"]] = choice_counts.get(result["choice"], 0) + 1

            for choice, count in choice_counts.items():
                if count >= majority_count:
                    self.state["final"] = {
                        "final_choice": choice,
                        "final_reasoning": "Consensus reached"
                    }
                    return self.state
        
        logger.info("No consensus reached after maximum consensus attempts among the panel. Proceeding to aggregation.")

        generic_agents_chat_history = {}
        for field_to_include in ["note", "problem", "static agents"]:
            generic_agents_chat_history[field_to_include] = self.state[field_to_include]
        aggregated_response = await self._aggregate(generic_agents_chat_history.copy())
        if aggregated_response is None:
            logger.error("Aggregation failed; returning error.")
        return self.state

class GenericAgent(LLMAgent):
    def __init__(self, agent_id: Union[str, int], state: dict):
        self.agent_id = str(agent_id)
        self.state = state
        self.state["static agents"][self.agent_id] = {} 

        self.round_id: int = 0
        self.schema = None
        system_prompt = "You are a collaborating diagnostic agent in a multi‑agent AI system designed to handle medical questions."

        super().__init__(system_prompt)

        logger.info(f"[GenericAgent '{self.agent_id}'] Initialized...")

    async def analyse_note(self, note: str, problem: str):
        self.round_id += 1
        self.schema = Response.model_json_schema()

        user_prompt = (
            "Read the patient note and decide whether the patient has the specified problem.\n"
            f"<<<PATIENT NOTE>>>\n{note}\n<<<END NOTE>>>\n\n"
            f"Question: Does this patient have {problem}?\n"
            "Please give your reasoning, then the choice ('Yes' or 'No')."
        )

        try:
            raw = await self.llm_call(
                user_prompt,
                temperature=0.1,
                guided_={"guided_json": self.schema}
            )
            parsed = safe_json_load(raw)
            self.append_message(raw)
            self.state["static agents"][self.agent_id][f"round_{self.round_id}"] = parsed
            return parsed
        except Exception as e:
            logger.error("[%s] analyse_note() failed: %s", self.agent_id, e)
            return None

    async def debate(self): 
        self.round_id += 1
        peers = {
            agent: info[f"round_{self.round_id-1}"]
            for agent, info in self.state["static agents"].items()
            if agent != self.agent_id
        }

        user_prompt = (
                    "Here are your peers’ previous answers:\n"
                    f"{json.dumps(peers, indent=2)}\n\n"
                    "Please review their reasoning. Based on their input and your own analysis, reconsider your initial assessment. "
                    "You may either keep your original conclusion or change it.\n\n"
                    "Please give your refined reasoning and final choice ('Yes' or 'No')."
                    )

        try:
            raw = await self.llm_call(
                user_prompt,
                temperature=0.1,
                guided_={"guided_json": self.schema}
            )
            parsed = safe_json_load(raw)
            self.append_message(raw)
            self.state["static agents"][self.agent_id][f"round_{self.round_id}"] = parsed
            return parsed
        except Exception as e:
            logger.error("[%s] debate() failed: %s", self.agent_id, e)
            return None
        

class DynamicSpecialist(LLMAgent):
    def __init__(self, specialist: str, panel_id: int, state: dict): 
        self.specialist = specialist
        self.state = state

        self.panel_id = panel_id
        self.expertise = state[f"panel_{self.panel_id}"]["Collected Specialists"][self.specialist]["expertise"]
        self.answer_history = state[f"panel_{self.panel_id}"]["Collected Specialists"][self.specialist]["answer_history"]
        self.round_id: int = 0
        self.schema = None
        system_prompt = (
            f"You are a {self.specialist}. "
            f"Your areas of expertise include:\n{self.expertise}\n"
            "Please analyze the patient's condition from the viewpoint of your specialty."
        )
        super().__init__(system_prompt)
        
        logger.info(f"[{self.specialist}] Initialized...")

    async def analyze_note(self, note: str, problem: str) -> Union[dict, None]: 
        self.round_id += 1


        self.schema = Response.model_json_schema()

        user_prompt = (
            "Read the patient note and decide whether the patient has the specified problem.\n"
            f"<<<PATIENT NOTE>>>\n{note}\n<<<END NOTE>>>\n\n"
            f"Question: Does this patient have {problem}?\n"
            "Please give your reasoning, then the choice ('Yes' or 'No')."
        )

        try:
            raw = await self.llm_call(
                user_prompt, 
                temperature=0.1, 
                guided_={"guided_json": self.schema}
            )
            parsed = safe_json_load(raw)
            self.append_message(content=raw)
            self.answer_history[f"round_{self.round_id}"] = parsed
            return parsed
        except Exception as e:
            logger.error(f"[{self.specialist}] analyze_note() LLM call failed: {e}")
            return None

    
    async def debate(self):
        self.round_id += 1
        other_specialists = {
            role: info["answer_history"][f"round_{self.round_id - 1}"]
            for role, info in self.state[f"panel_{self.panel_id}"]["Collected Specialists"].items()
            if role != self.specialist
        }

        user_prompt = (
                    "Here are other specialists' previous answers:\n"
                    f"{json.dumps(other_specialists, indent=2)}\n\n"
                    "Please review their reasoning. Based on their input and your own analysis, reconsider your initial assessment. "
                    "You may either keep your original conclusion or change it.\n\n"
                    "Please give your refined reasoning and final choice ('Yes' or 'No')."
                    )

        try:
            raw = await self.llm_call(user_prompt, temperature=0.1, guided_={"guided_json": self.schema})
            parsed = safe_json_load(raw)
            self.append_message(raw)
            self.answer_history[f"round_{self.round_id}"] = parsed
            return parsed
        except Exception as e:
            logger.error(f"[{self.specialist}] debate() LLM call failed: {e}")
            return None


class CaseBasedAgent(LLMAgent):
    def __init__(self, state_dict: dict):
        self.state_dict = state_dict
        system_prompt = (
            "You are a Case-based RAG specialist. "
            "Your role is to compare the current patient's note against a large collection of clinical notes,\n"
            "retrieve similar cases and their discharge diagnoses, and then infer whether the patient likely\n"
            "has the specified problem. Provide a final choice ('Yes' or 'No') and your reasoning."
        )
        super().__init__(system_prompt)

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        self.tokenizer = AutoTokenizer.from_pretrained(
            "nvidia/NV-Embed-v2",
            trust_remote_code=True,
            cache_dir="/secure/shared_data/rag_embedding_model"
        )
        self.embedding_model = AutoModel.from_pretrained(
            "nvidia/NV-Embed-v2",
            trust_remote_code=True,
            cache_dir="/secure/shared_data/rag_embedding_model",
            device_map="auto"
        )

        with open("/secure/shared_data/SOAP/MIMIC/cases_base.json", "r") as f:
            self.cases = json.load(f)
        self.docs = create_documents(self.cases, self.tokenizer, max_length=512)

        self.db_client = chromadb.PersistentClient(
            path="/secure/shared_data/rag_embedding_model/chroma_db",
            settings=Settings(allow_reset=True)
        )

        logger.info("[CaseBasedAgent] Initialized.")

    def _retrieve_cases(self, query_note: str, collection_name: str = "mimic_notes_full"):
        collection = self.db_client.get_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        query_prefix = (
            "Given the following clinical note, retrieve the most similar clinical case. "
            "The clinical note is:\n\n"
        )
        retrieved = hybrid_query(
            cases=self.cases,
            docs=self.docs,
            collection=collection,
            embedding_model=self.embedding_model,
            query_text=query_note,
            query_prefix=query_prefix,
            max_length=512,
            semantic_k=5,
            bm25_k=5,
            bm25_weight=0.5
        )
        
        return retrieved

    async def analyze_note(self, note: str, problem: str):
        retrieved_cases = self._retrieve_cases(note)

        retrieved_text = ""
        for i, case in enumerate(retrieved_cases, start=1):
            case_str = (
                f"<<<START RETRIEVED CASE #{i}>>>\n"
                f"Excerpt: {case['text']}\n"
                f"Discharge Diagnosis: {case['diagnosis']}\n"
                f"Similarity Score: {case['score']}\n"
                f"<<<END RETRIEVED CASE #{i}>>>\n\n"
            )
            if count_llama_tokens(note + retrieved_text + case_str) < 18000:
                retrieved_text += case_str
            else:
                logger.warning("[CaseBasedAgent] Retrieved text exceeds token limit, truncating.")
                break

        user_prompt = (
            "Please read the following patient note:\n\n"
            "<<<START NOTE>>>\n"
            f"{note}\n"
            "<<<END NOTE>>>\n\n"
            "We compared this note against a large knowledge base of clinical notes.\n"
            "Here are some retrieved examples:\n\n"
            "<<<START RETRIEVED EXCERPTS>>>\n"
            f"{retrieved_text}"
            "<<<END RETRIEVED EXCERPTS>>>\n\n"
            "Now, you must decide if the patient likely has the following problem:\n\n"
            f"<<<PROBLEM>>>\n{problem}\n<<<END PROBLEM>>>\n\n"
            "### Your Task ###\n"
            "1) Summarize any relevant similarities and differences between the retrieved cases and the current note.\n"
            "2) Provide your reasoning for the final choice.\n"
            "3) Provide your final choice ('Yes' or 'No').\n"
        )

        class RagResponse(BaseModel):
            summary: str = Field(..., description="Summary of similarities and differences.")
            reasoning: str = Field(..., description="Step-by-step reasoning leading to the final choice.")
            choice: Literal["Yes", "No"] = Field(..., description=f"Final choice indicating whether the patient has {problem}.")

        guided_schema = RagResponse.model_json_schema()

        try:
            response = await self.llm_call(
                user_prompt, 
                guided_={"guided_json": guided_schema}
            )
        except Exception as e:
            logger.error(f"[CaseBasedAgent] analyze_note() LLM call failed: {e}")
            return None

        self.append_message(content=response)
        parsed_response = safe_json_load(response)
        if parsed_response:
            summary = parsed_response.get("summary", "")
            reasoning = parsed_response.get("reasoning", "")
            choice = parsed_response.get("choice", "Error in choice.")
            self.state_dict.update({
                "CaseBasedAgent": {
                    "retrieved cases": retrieved_text, 
                    "summary": summary, 
                    "reasoning": reasoning, 
                    "choice": choice
                }
            })
        return parsed_response


async def run_generic(note, hadm_id, problem, label):
    mgr = Manager(note, hadm_id, problem, label,
                  n_generic_agents=5, consensus_threshold=0.8,
                  max_consensus_attempts=4, max_assignment_attempts=3)
    state = await mgr.run_generic_agents()
    return {
        "method": "generic_multi",
        "hadm_id": hadm_id,
        "label": label,
        "choice": state["final"]["final_choice"],
        "reasoning": state["final"]["final_reasoning"],
        "raw_state": state,          # keep everything if you want
    }

async def run_dynamic(note, hadm_id, problem, label):
    mgr = Manager(note, hadm_id, problem, label,
                  n_specialists="auto", consensus_threshold=0.8,
                  max_consensus_attempts=4, max_assignment_attempts=3)
    state = await mgr.run_specialists()
    return {
        "method": "dynamic_multi",
        "hadm_id": hadm_id,
        "label": label,
        "choice": state["final"]["final_choice"],
        "reasoning": state["final"]["final_reasoning"],
        "raw_state": state,
    }

async def run_baseline(note, hadm_id, problem, label):
    zs = BaselineZS()
    out = await zs.analyze_note(note, problem)
    if out:
        return {
            "method": "baseline_zs",
            "hadm_id": hadm_id,
            "label": label,
            "choice": out["choice"],
            "reasoning": out["reasoning"],
            "raw_state": out,
        }
    else:
        return {"method": "baseline_zs",
                "hadm_id": hadm_id,
                "label": label,
                "choice": "ERROR",
                "reasoning": "baseline failed",
                "raw_state": {}}


async def process_row(row, problem):
    note = f"{row['Subjective']}\n{row['Objective']}"
    hadm_id = row["File ID"]
    label = row["combined_summary"]

    tasks = [
        run_generic(note, hadm_id, problem, label),
        run_dynamic(note, hadm_id, problem, label),
        run_baseline(note, hadm_id, problem, label),
    ]
    return await asyncio.gather(*tasks)           # returns a list of 3 dicts

async def process_problem(df, problem):
    logger.info("Processing %s (%d rows).", problem, len(df))
    all_results = []

    for _, row in df.iterrows():
        all_results.extend(await process_row(row, problem))

    out_path = f"/home/yl3427/cylab/SOAP_MA/Output/SOAP/generic/integrated_results_{problem.replace(' ','_')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("[%s] saved to %s", problem, out_path)

async def main():
    df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/Input/SOAP_all_problems.csv", lineterminator="\n")
    tasks = [asyncio.create_task(process_problem(df, p))
             for p in selected_problems]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler('log/0421_MA_aki_integrated_test.log', mode='w'), # Save to file
            logging.StreamHandler()  # Print to console
        ]
    )

    asyncio.run(main())