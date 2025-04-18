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
    'congestive heart failure',
    'sepsis',
    'acute kidney injury',
]

class LLMAgent:
    def __init__(
        self, 
        system_prompt: str, 
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        client=AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy"),
        max_tokens: int = LLAMA3_70B_MAX_TOKENS,
        summarization_threshold: float = 0.75
    ):
        self.model_name = model_name
        self.client = client
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_tokens = max_tokens
        self.token_threshold = int(self.max_tokens * summarization_threshold)
        logger.info(f"[{self.__class__.__name__}] Initialized. Token limit: {self.max_tokens}, Summarization threshold: {self.token_threshold}")
        self._log_memory_size("Initialization")
    
    def _log_memory_size(self, context_message: str):
        """Helper to log current token count."""
        current_tokens = count_llama_tokens(self.messages)
        logger.info(f"[{self.__class__.__name__}] Memory: {current_tokens} tokens. (at: {context_message})")
        if current_tokens > self.token_threshold:
             logger.warning(f"[{self.__class__.__name__}] Token count ({current_tokens}) exceeds threshold ({self.token_threshold})!")
        if current_tokens > self.max_tokens:
             logger.error(f"[{self.__class__.__name__}] CRITICAL: Token count ({current_tokens}) exceeds model limit ({self.max_tokens})!")

    async def _summarize_history(self) -> bool: # inplace
        logger.warning("[%s] Starting iterative summarisation …", self.__class__.__name__)

        # helper -----------------------------------------------------------
        def longest_msg_idx() -> int:
            lengths = [
                (i, count_llama_tokens([msg]))    # token count of single msg
                for i, msg in enumerate(self.messages)
                if msg["role"] != "system"        # never summarise system prompt(s)
            ]
            return max(lengths, key=lambda t: t[1])[0]

        # guard — if there is only the system prompt + 1 msg, bail ----------
        if len(self.messages) < 3:
            logger.warning("Not enough messages to summarise.")
            return False

        any_change = False
        while count_llama_tokens(self.messages) >= self.token_threshold:
            idx = longest_msg_idx()
            target_msg = self.messages[idx]["content"]

            summary_prompt = (
                "Summarise the following message as concisely as possible, "
                "preserving all key facts and reasoning steps. "
                "Use no more than 1000 words.\n\n"
                "<<<MESSAGE_START>>>\n"
                f"{target_msg}\n"
                "<<<MESSAGE_END>>>"
            )

            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a summarisation assistant."},
                        {"role": "user",   "content": summary_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1500,
                )
                new_text = resp.choices[0].message.content.strip()

            except Exception as e:
                logger.error("Summarisation call failed: %s", e)
                return False

            # if the summary is *longer* (unlikely but possible), abort loop
            if count_llama_tokens([{"role": "assistant", "content": new_text}]) >= \
            count_llama_tokens([self.messages[idx]]):
                logger.warning("Summary did not reduce length; aborting.")
                break

            # replace the message ------------------------------------------
            self.messages[idx]["content"] = f"[Summary] {new_text}"
            any_change = True
            logger.info("Replaced longest message with summary (%d tokens now).",
                        count_llama_tokens(self.messages))

        if any_change:
            self._log_memory_size("After iterative summarisation")
        return any_change


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
        temperature: float = 0.5,
        guided_: dict = None,
        tools_descript: List[dict] = None, 
        available_tools: dict = None
    ) -> Any:
        logger.debug(f"LLMAgent.llm_call() - user_prompt[:60]: {user_prompt[:60]}...")

        # Check BEFORE appending the new user prompt
        potential_new_length = count_llama_tokens(self.messages + [{"role": "user", "content": user_prompt}])
        if potential_new_length > self.token_threshold:
             summarized = await self._summarize_history()
             if not summarized and potential_new_length > self.max_tokens:
                  logger.error("History too long and summarization failed. Aborting call.")
                  raise ValueError("Context limit exceeded even after summarization attempt.")
        
        self.messages.append({"role": "user", "content": user_prompt})
        self._log_memory_size("After appending user prompt")

        params = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": temperature,
        }
        if guided_:
            logger.debug(f"Guided JSON/choice detected: {guided_}")
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
        self._log_memory_size(f"After explicit append_message (role={role})")
        # We won't trigger summarization here, but rely on checks before LLM calls.
        # If append causes it to exceed max_tokens, the next LLM call will fail/summarize.

class Manager(LLMAgent):
    def __init__(
        self, 
        note: str, 
        hadm_id: str, 
        problem: str, 
        label: str, 
        n_specialists: Union[int, Literal["auto"]] = 'auto',
        static_agents: Optional[List[str]] = None,     # NEW: list of system prompts
        consensus_threshold: float = 0.8,
        max_consensus_attempts=3, 
        max_assignment_attempts=2,
        static_specialists: Optional[List[object]] = None
    ):

        system_prompt = (
            "You are a Manager agent in a multi-agent AI system designed to handle medical questions.\n"
            "Your job is to choose a set of medical specialists whose expertise is relevant to the user's query\n"
            "and then ensure they reach a consensus on the correct answer.\n"
        )
        super().__init__(system_prompt)

        self.note = note
        self.hadm_id = hadm_id
        self.problem = problem
        self.label = label
        
        self.state_dict = {
            "note": note, 
            "hadm_id": hadm_id, 
            "problem": problem, 
            "label": label,
            "cached_messages": [],
        }
        self.n_specialists = n_specialists
        self.consensus_threshold = consensus_threshold
    
        self.max_consensus_attempts = max_consensus_attempts
        self.consensus_attempts = 0  # round

        self.max_assignment_attempts = max_assignment_attempts
        self.assignment_attempts = 0  # panel

        self.static_specialists = static_specialists
    
    async def _assign_specialists(self):
        self.assignment_attempts += 1
        logger.info(f"Starting assignment attempt #{self.assignment_attempts}.")
        self.state_dict[f"panel_{self.assignment_attempts}"] = {}

        if self.assignment_attempts > 1:
            previous_panel = self.state_dict[f"panel_{self.assignment_attempts - 1}"]["Initially Identified Specialties"]
            self._cache_and_clear_memory(initial_message=(f"The previous panel of specialists {previous_panel} couldn’t reach consensus, "
                                                          f"so we’re now assembling a new panel."
                                                          ))
            logger.info(f"Cleared memory and cached messages for the previous assignment attempt.")

        specialties_lst = []

        trial_n = 0
        while specialties_lst == [] and trial_n < 3:
            if self.n_specialists == "auto":
                user_prompt = (
                    "Below are the Subjective (S) and Objective (O) sections from a SOAP note:\n\n"
                    f"<SOAP>\n{self.note}\n</SOAP>\n\n"
                    f"We need to determine if this patient has {self.problem}.\n\n"
                    "Please list the medical specialties that should be consulted to make a reliable diagnosis."
                )
                class Specialties(BaseModel):
                    specialties: List[str] = Field(..., description="List of medical specialties needed to address the case.")
                try:
                    response = await self.llm_call(user_prompt, guided_={"guided_json": Specialties.model_json_schema()})
                    self.append_message(content=response)
                    data = safe_json_load(response)
                    specialties_lst = data["specialties"]
                except Exception as e:
                    logger.error(f"Failed to parse 'specialties' from LLM response: {e}")

                logger.debug(f"{len(specialties_lst)} specialties identified via 'auto': {specialties_lst}")
                self.n_specialists = len(specialties_lst)

            else:
                user_prompt = (
                    "Below are the Subjective (S) and Objective (O) sections from a SOAP note:\n\n"
                    f"<SOAP>\n{self.note}\n</SOAP>\n\n"
                    f"We need to evaluate whether the patient has {self.problem}.\n\n"
                    f"Please provide a list of {self.n_specialists} medical specialties that should be involved.\n"
                    "These specialties must collectively cover all aspects needed to diagnose this case."
                )
                class Specialty(BaseModel):
                    name: str = Field(..., description="Name of the medical specialty required.")
                specialties_dict = {f"specialty_{i+1}": (Specialty, ...) for i in range(self.n_specialists)}
                Specialties_N = create_model("Specialties_N", **specialties_dict)
                try:
                    response = await self.llm_call(user_prompt, guided_={"guided_json": Specialties_N.model_json_schema()})
                    self.append_message(content=response)
                    parsed = safe_json_load(response)
                except Exception as e:
                    logger.error(f"Failed to parse specialties from LLM response: {e}")
                    parsed = {}
                try:
                    for i in range(self.n_specialists):
                        # Potential KeyError if "specialty_i" or 'name' is missing
                        name = parsed[f"specialty_{i+1}"]["name"]
                        specialties_lst.append(name)
                except Exception as e:
                    logger.error(f"Failed to parse specialty from LLM response: {e}")
            
            trial_n += 1

        self.state_dict[f"panel_{self.assignment_attempts}"]["Initially Identified Specialties"] = specialties_lst # when something goes wrong, this is empty
        self.state_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"] = {}

        # Request a panel of specialists for each identified specialty
        user_prompt = (
            "Based on the list of specialties you provided: "
            f"{specialties_lst}\n\n"
            f"Please assemble a panel of {self.n_specialists} specialists. Assign each specialist to exactly one of the above specialties.\n"
            "For each specialist, specify their role (not their personal name) and list the relevant expertise areas related to this case."
        )

        class Specialist(BaseModel):
            specialist: str = Field(..., description="Official job title. No personal names.")
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
             
                self.state_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"][role] = {
                    "expertise": expertise, 
                    "answer_history": {}
                }
                logger.info(f"Specialist {role} assigned with expertise: {expertise}")
        except Exception as e:
            logger.error(f"Failed to parse specialists from LLM response: {e}")
            return None
        
        return self.state_dict
    
    def _check_consensus(self, panel_id: int, round_id: int) -> Optional[str]:
        logger.info(f"Checking for consensus among specialists in panel_{panel_id}, round_{round_id}.")
        choice_counts = {}
        majority_count = math.ceil(self.n_specialists * self.consensus_threshold)

        for role, answ_hist in self.state_dict[f"panel_{panel_id}"]["Collected Specialists"].items():
            final_choice = answ_hist["answer_history"][f"round_{round_id}"]['choice']
            choice_counts[final_choice] = choice_counts.get(final_choice, 0) + 1

        for choice, count in choice_counts.items():
            if count >= majority_count:
                logger.info(f"Consensus found on choice '{choice}' with {count}/{self.n_specialists} specialists.")
                return choice
        logger.info("No consensus found.")
        return None

    async def _aggregate(self):
        specialists_chat_history = self.state_dict.copy()
        for field_to_remove in ["label", "hadm_id", "problem", "cached_messages"]:
            specialists_chat_history.pop(field_to_remove, None)

        specialists_str = json.dumps(specialists_chat_history, indent=4)
 
        logger.info(f"Token count for specialists' chat history: {count_llama_tokens(specialists_str)}")

        user_prompt = (
            "No consensus has been reached among the specialists.\n"
            "You now have access to the entire conversation histories for all specialists.\n"
            "Please analyze each specialist's reasoning and final choice, then provide a single, definitive answer.\n"
            "Your answer should be the one best supported by their collective reasoning.\n\n"
            "Below is the complete conversation history for each specialist:\n\n"
            f"{specialists_str}\n\n"
            "After reviewing this material, please provide:\n"
            "1) A concise summary of the reasoning behind your final decision\n"
            "2) A single recommended choice: 'Yes' or 'No' "
            f"(indicating whether the patient has {self.problem})."
        )

        class AggregatedResponse(BaseModel):
            aggregated_reasoning: str = Field(..., description="Detailed reasoning behind the final choice.")
            aggregated_choice: Literal["Yes", "No"] = Field(
                ..., description=f"Single recommended choice for whether the patient has {self.problem}."
            )
        
        try:
            response = await self.llm_call(user_prompt, temperature=0.1, guided_={"guided_json": AggregatedResponse.model_json_schema()})
            data = safe_json_load(response)
        except Exception as e:
            logger.error(f"Failed to parse aggregator response: {e}")
            data = {}
        return data
    
    def _cache_and_clear_memory(self, initial_message: str = ""):
        self.state_dict["cached_messages"].extend(self.messages.copy())
        self.messages = [self.messages[0]]  # Keep only the system message
        if initial_message:
            self.messages.append({"role": "system", "content": f"Summary so far: {initial_message}"})
        logger.info(f"[{self.__class__.__name__}] Cached messages and cleared memory. Current token length: {count_llama_tokens(self.messages)}")

    async def run(self):
        consecutive_failures = 0
        while self.assignment_attempts < self.max_assignment_attempts:
            logger.info(f"Assignment attempt #{self.assignment_attempts + 1} started.")
            
            self.consensus_attempts = 0
            panel_state = await self._assign_specialists()  # increments self.assignment_attempts
            # 굳이 안반환해도 됨
            if panel_state is None:
                logger.error("Failed to assign specialists; retrying.")
                consecutive_failures += 1
                if consecutive_failures >= self.max_assignment_attempts:
                    break                      # fall through to aggregation
                continue
            consecutive_failures = 0
          
            panel = []
            for role in self.state_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"].keys():
                panel.append(
                    DynamicSpecialist(role, self.state_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"][role])
                )
                
            # Step 1) Specialists analyze the note
            analyze_tasks = [asyncio.create_task(specialist.analyze_note(self.note, self.problem)) for specialist in panel]
            analyze_results = await asyncio.gather(*analyze_tasks)
            if any(r is None for r in analyze_results):
                logger.error("At least one specialist failed analysis; skipping this panel.")
                continue


            # Check immediate consensus (round 1)
            self.consensus_attempts += 1
            consensus_choice = self._check_consensus(self.assignment_attempts, self.consensus_attempts)
            if consensus_choice:
                self.state_dict["final"] = {
                    "final_choice": consensus_choice, 
                    "final_reasoning": "Consensus reached"
                }
                return self.state_dict
            
            # Debate loop
            while self.consensus_attempts < self.max_consensus_attempts:
                logger.info(f"Debate attempt #{self.consensus_attempts + 1} started.")
                debate_tasks = [
                    asyncio.create_task(
                        specialist.debate(self.state_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"])
                    ) 
                    for specialist in panel
                ]
                debate_results = await asyncio.gather(*debate_tasks)
                if any(r is None for r in debate_results):
                    logger.error("At least one specialist failed during debate; skipping this round.")
                    continue

                self.consensus_attempts += 1
                consensus_choice = self._check_consensus(self.assignment_attempts, self.consensus_attempts)
                if consensus_choice:
                    self.state_dict["final"] = {
                        "final_choice": consensus_choice, 
                        "final_reasoning": "Consensus reached"
                    }
                    return self.state_dict
            
            # If we exhaust all debate attempts with no consensus, summarize the debate, then move on
            logger.info("No consensus reached after maximum consensus attempts among the panel.")

        # If we reach here, we've exhausted assignment attempts, so we do final aggregation:
        logger.info("No consensus reached after maximum assignment attempts. Proceeding to aggregation.")
        aggregated_response = await self._aggregate()
        self.state_dict["final"] = {
            "final_choice": aggregated_response.get("aggregated_choice", "Error in aggregation."),
            "final_reasoning": aggregated_response.get("aggregated_reasoning", "Unable to parse aggregator response.")
        }
        return self.state_dict


class DynamicSpecialist(LLMAgent):
    def __init__(self, specialist: str, status: dict): 
        self.specialist = specialist
        # status expected: {"expertise": [...], "answer_history": {...}}
        self.expertise = status["expertise"]
        self.answer_history = status["answer_history"]
        self.round_id = 0
        self.schema = None
        system_prompt = (
            f"You are a {self.specialist}. "
            f"Your areas of expertise include:\n{self.expertise}\n"
            "Please analyze the patient's condition from the viewpoint of your specialty."
        )
        super().__init__(system_prompt)
        
        logger.info(f"[{self.specialist}] Initialized...")

    async def analyze_note(self, note: str, problem: str):
        self.round_id += 1

        class Response(BaseModel):
            reasoning: str = Field(..., description="Step-by-step reasoning leading to the final choice.")
            choice: Literal["Yes", "No"] = Field(..., description=f"Final choice indicating whether the patient has {problem}.")
        self.schema = Response.model_json_schema()

        user_prompt = (
            "Here are the Subjective (S) and Objective (O) parts of a SOAP note:\n\n"
            f"<SOAP>\n{note}\n</SOAP>\n\n"
            "Based on this information, does the patient have the following problem?\n\n"
            f"<Problem>\n{problem}\n</Problem>\n\n"
            f"Acting as a {self.specialist}, please provide:\n"
            "1. Your step-by-step reasoning.\n"
            "2. Your choice: 'Yes' or 'No'."
        )

        try:
            response = await self.llm_call(user_prompt, temperature=0.1, guided_={"guided_json": self.schema})
            parsed_response = safe_json_load(response)
            self.append_message(content=response)
            self.answer_history[f"round_{self.round_id}"] = parsed_response
            return parsed_response
        
        except Exception as e:
            logger.error(f"[{self.specialist}] analyze_note() LLM call failed: {e}")
            return None

    
    async def debate(self, stepback_status: dict):
        self.round_id += 1
        other_specialists = {}
        for role, value in stepback_status.items():
            if role != self.specialist:
                try:
                    other_specialists[role] = value["answer_history"][f"round_{self.round_id - 1}"]
                except KeyError as e:
                    logger.error(f"[{self.specialist}] Missing data from {role}: {e}")
                    continue

        formatted_other_specialists = json.dumps(other_specialists, indent=4)
        user_prompt = (
            "You have received opinions from other specialists:\n\n"
            f"{formatted_other_specialists}\n\n"
            "Please review the reasoning of the other specialists. Based on their input and your own analysis, reconsider your initial assessment. "
            "You can either stick with your original conclusion or change it.\n\n"
            "Provide the following:\n"
            "1. Your final reasoning.\n"
            "2. Your final choice: 'Yes' or 'No'."
        )

        try:
            response = await self.llm_call(user_prompt, temperature=0.3, guided_={"guided_json": self.schema})
            parsed_response = safe_json_load(response)
            self.append_message(content=response)
            self.answer_history[f"round_{self.round_id}"] = parsed_response
            return parsed_response
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

        class Response(BaseModel):
            summary: str = Field(..., description="Summary of similarities and differences.")
            reasoning: str = Field(..., description="Step-by-step reasoning leading to the final choice.")
            choice: Literal["Yes", "No"] = Field(..., description=f"Final choice indicating whether the patient has {problem}.")

        guided_schema = Response.model_json_schema()

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


async def process_problem(df: pd.DataFrame, problem: str):
    logger.info(f"Processing problem '{problem}' for {len(df)} rows.")
    results = []

    for idx, row in df.iterrows():
        logger.info(f"[{problem}] Processing row index {idx}")

        note_text = str(row["Subjective"]) + "\n" + str(row['Objective'])
        hadm_id = row["File ID"]
        label = row["combined_summary"]

        manager = Manager(
            note=note_text,
            hadm_id=hadm_id,
            problem=problem,
            label=label,
            n_specialists='auto',  # or an integer
            consensus_threshold=0.8,
            max_consensus_attempts=4,
            max_assignment_attempts=3,
            static_specialists=None,
        )

        # Run the manager's workflow
        result = await manager.run()
        results.append(result)

    # Save results for this problem
    output_path = f"/home/yl3427/cylab/SOAP_MA/Output/SOAP/correction_all_3/3_problems_{problem.replace(' ', '_')}_new_temp.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"[{problem}] Results saved to: {output_path}")

async def main():
    df_path = "/home/yl3427/cylab/SOAP_MA/Input/SOAP_all_problems.csv"
    df = pd.read_csv(df_path, lineterminator='\n')
    logger.info("Loaded dataframe with %d rows.", len(df))

    # Create an asyncio Task for each problem
    tasks = []
    for problem in selected_problems:
        tasks.append(asyncio.create_task(process_problem(df, problem)))

    # Run them concurrently
    await asyncio.gather(*tasks)

    logger.info("All tasks completed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler('log/0417_MA_3_probs_outof_all_corrected.log', mode='w'), # Save to file
            logging.StreamHandler()  # Print to console
        ]
    )

    asyncio.run(main())
