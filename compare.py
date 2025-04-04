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
import math

logger = logging.getLogger(__name__)


def safe_json_load(s: str) -> any:
    """
    Attempts to parse a JSON string using multiple parsers.
    Order:
    1. json.loads (strict)
    2. demjson3.decode (tolerant)
    3. json5.loads (allows single quotes, unquoted keys, etc.)
    4. dirtyjson.loads (for messy JSON)
    5. jsom (if available)
    6. json_repair (attempt to repair the JSON and parse it)
    
    If all attempts fail, returns None.
    """
    # 1. Try standard JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        logger.error("Standard json.loads failed: %s", e)
    
    # 2. Try demjson3
    try:
        import demjson3
        logger.info("Attempting to parse with demjson3 as fallback.")
        result = demjson3.decode(s)
        logger.info("demjson3 successfully parsed the JSON.")
        return result
    except Exception as e2:
        logger.error("demjson3 fallback failed: %s", e2)
    
    # 3. Try json5
    try:
        import json5
        logger.info("Attempting to parse with json5 as fallback.")
        result = json5.loads(s)
        logger.info("json5 successfully parsed the JSON.")
        return result
    except Exception as e3:
        logger.error("json5 fallback failed: %s", e3)
    
    # 4. Try dirtyjson
    try:
        import dirtyjson
        logger.info("Attempting to parse with dirtyjson as fallback.")
        result = dirtyjson.loads(s)
        logger.info("dirtyjson successfully parsed the JSON.")
        return result
    except Exception as e4:
        logger.error("dirtyjson fallback failed: %s", e4)
    
    # 5. Try jsom
    try:
        import jsom
        logger.info("Attempting to parse with jsom as fallback.")
        parser = jsom.JsomParser()
        result = parser.loads(s)
        logger.info("jsom successfully parsed the JSON.")
        return result
    except Exception as e5:
        logger.error("jsom fallback failed: %s", e5)
    
    # 6. Try json_repair (attempt to fix the JSON and then load it)
    try:
        import json_repair
        logger.info("Attempting to repair JSON with json_repair as fallback.")
        repaired = json_repair.repair_json(s)
        result = json.loads(repaired)
        logger.info("json_repair successfully parsed the JSON.")
        return result
    except Exception as e6:
        logger.error("json_repair fallback failed: %s", e6)
    
    # All attempts failed; return None
    logger.error("All JSON parsing attempts failed. Returning None.")
    logger.error("Original input: %s", s)
    return None


class LLMAgent:
    def __init__(
        self, 
        system_prompt: str, 
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        client=AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    ):
        self.model_name = model_name
        self.client = client
        self.messages = [{"role": "system", "content": system_prompt}]
        logger.info(f"[{self.__class__.__name__}] Initialized with memory length: {len(self.messages)}")  # <-- Logging initial memory size
    
    async def __llm_call_with_tools(self, params: dict, available_tools: dict) -> Any:
        iter_count = 0
        
        try:
            response = await self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error(f"Error calling LLM with tools: {e}")
            return None
        
        message = response.choices[0].message
          
        while not hasattr(message, "tool_calls") or not message.tool_calls:
            logger.info("No tool calls detected in response.")
            
            # Add the assistant's textual content
            self.messages.append({"role": "assistant", "content": message.content})
            logger.info(f"[{self.__class__.__name__}] memory length after append: {len(self.messages)}")  # <-- Logging memory size
            
            # Provide a hint/warning
            self.messages.append({"role": "system", "content": "WARNING! Remember to call a tool."})
            logger.info(f"[{self.__class__.__name__}] memory length after append: {len(self.messages)}")  # <-- Logging memory size

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
        # Append the calls
        self.messages.append({"role": "assistant", "tool_calls": message.tool_calls})
        logger.info(f"[{self.__class__.__name__}] memory length after append: {len(self.messages)}")  # <-- Logging memory size
        
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
                continue

            try:
                # Synchronous call to tool
                result = available_tools[call.function.name](**args)
            except Exception as e:
                logger.error(f"Error calling tool '{call.function.name}': {e}")
                return message.content
            
            self.messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": call.id,
                "name": call.function.name,
            })
            logger.info(f"[{self.__class__.__name__}] memory length after append: {len(self.messages)}")  # <-- Logging memory size
        
        try:
            response = await self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error(f"Error calling LLM after tool usage: {e}")
            return None

        return response.choices[0].message.content

    async def llm_call(
        self, 
        user_prompt: str, 
        temperature: float = 0.3,
        guided_: dict = None,
        tools_descript: List[dict] = None, 
        available_tools: dict = None
    ) -> Any:
        logger.debug(f"LLMAgent.llm_call() - user_prompt[:60]: {user_prompt[:60]}...")
        
        # Append the user's prompt
        self.messages.append({"role": "user", "content": user_prompt})
        logger.info(f"[{self.__class__.__name__}] memory length after append: {len(self.messages)}")  # <-- Logging memory size
        
        params = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": temperature,
        }
        if guided_:
            logger.debug(f"Guided JSON/choice detected: {guided_}")
            params["extra_body"] = guided_

        # If we have tools described, handle it via __llm_call_with_tools
        if tools_descript:
            params["tools"] = tools_descript
            assert available_tools is not None, "available_tools must be provided if tools_descript is used."
            return await self.__llm_call_with_tools(params, available_tools)
        else:
            # No tool usage
            params["tool_choice"] = "none"
            try:
                response = await self.client.chat.completions.create(**params)
            except Exception as e:
                logger.error(f"Error calling LLM in llm_call(): {e}")
                return None

            logger.debug("LLM call succeeded, returning response content.")
            return response.choices[0].message.content
    
    def append_message(self, content, role='assistant'):
        """
        Helper to append a new message to this agent's conversation and log memory length.
        """
        logger.debug(f"Appending message with role='{role}' to conversation.")
        self.messages.append({"role": role, "content": content})
        # Log how large the conversation is now:
        logger.info(f"[{self.__class__.__name__}] memory length after append: {len(self.messages)}")


class Manager(LLMAgent):
    def __init__(
        self, 
        note: str, 
        hadm_id: str, 
        problem: str, 
        label: Literal["Yes", "No"], 
        n_specialists: Union[int, Literal["auto"]] = 5,  
        consensus_threshold: float = 0.8,
        max_consensus_attempts=3, 
        max_assignment_attempts=1,
        static_specialists: Optional[List[object]] = None, 
        summarizer: Dict[Literal["before_manager", "before_specialist"], Callable[..., any]] = None
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
        
        self.status_dict = {
            "note": note, 
            "hadm_id": hadm_id, 
            "problem": problem, 
            "label": label
        }
        self.n_specialists = n_specialists
        self.consensus_threshold = consensus_threshold
    
        self.max_consensus_attempts = max_consensus_attempts
        self.consensus_attempts = 0  # round

        self.max_assignment_attempts = max_assignment_attempts
        self.assignment_attempts = 0  # panel

        self.static_specialists = static_specialists
        self.summarizer = summarizer
    
    async def _assign_specialists(self):
        self.assignment_attempts += 1
        logger.info(f"Starting assignment attempt #{self.assignment_attempts}.")
        self.status_dict[f"panel_{self.assignment_attempts}"] = {}

        if self.n_specialists == "auto":
            user_prompt = (
                "Below are the Subjective (S) and Objective (O) sections from a SOAP note:\n\n"
                f"<SOAP>\n{self.note}\n</SOAP>\n\n"
                f"We need to determine if this patient has {self.problem}.\n\n"
                "Please list the medical specialties that should be consulted to make a reliable diagnosis."
            )

            class Specialties(BaseModel):
                specialties: List[str] = Field(..., description="List of medical specialties needed to address the case.")

            response = await self.llm_call(user_prompt, guided_={"guided_json": Specialties.model_json_schema()})
            self.append_message(content=response)

            try:
                data = safe_json_load(response)
                specialties_lst = data["specialties"]
            except KeyError as e:
                logger.error(f"KeyError parsing 'specialties' from LLM response: {e}")
                specialties_lst = []
            except Exception as e:
                logger.error(f"Failed to parse 'specialties' from LLM response: {e}")
                specialties_lst = []

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

            response = await self.llm_call(user_prompt, guided_={"guided_json": Specialties_N.model_json_schema()})
            self.append_message(content=response)

            specialties_lst = []
            parsed = safe_json_load(response)
            try:
                for i in range(self.n_specialists):
                    name = parsed[f"specialty_{i+1}"]["name"]
                    specialties_lst.append(name)
            except KeyError as e:
                logger.error(f"KeyError parsing 'specialty_{i+1}' from LLM response: {e}")
            except Exception as e:
                logger.error(f"Failed to parse specialty from LLM response: {e}")

        self.status_dict[f"panel_{self.assignment_attempts}"]["Initially Identified Specialties"] = specialties_lst
        self.status_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"] = {}

        # Request a panel of specialists for each identified specialty
        user_prompt = (
            "Based on the list of specialties you provided:\n"
            f"{specialties_lst}\n\n"
            f"Please assemble a panel of {self.n_specialists} specialists. Assign each specialist to exactly one "
            "of the above specialties. For each specialist, please provide:\n"
            "1) Their official title (role)\n"
            "2) A list of relevant expertise areas for this case.\n"
        )
        class Specialist(BaseModel):
            specialist: str = Field(..., description="Official job title.")
            expertise: List[str] = Field(..., description="Their main areas of expertise relevant to this case.")

        panel_dict = {f"specialist_{i+1}": (Specialist, ...) for i in range(self.n_specialists)}
        SpecialistPanel = create_model("SpecialistPanel", **panel_dict)

        panel_response = await self.llm_call(user_prompt, guided_={"guided_json": SpecialistPanel.model_json_schema()})
        self.append_message(content=panel_response)

        specialists_dict = safe_json_load(panel_response)
        if not specialists_dict:
            logger.error(f"Failed to parse the specialist panel: {panel_response}")
            specialists_dict = {}

        for key, specialist in specialists_dict.items():
            try:
                role = specialist["specialist"]
                expertise = specialist["expertise"]
            except KeyError as e:
                logger.error(f"KeyError in specialist data: {e}")
                continue

            self.status_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"][role] = {
                "expertise": expertise, 
                "answer_history": {}
            }
        return self.status_dict
    
    def _check_consensus(self, panel_id: int, round_id: int) -> Optional[str]:
        choice_counts = {}
        majority_count = math.ceil(self.n_specialists * self.consensus_threshold)

        for role, answ_hist in self.status_dict[f"panel_{panel_id}"]["Collected Specialists"].items():
            try:
                final_choice = answ_hist["answer_history"][f"round_{round_id}"]['choice']
            except KeyError as e:
                logger.error(f"KeyError reading final_choice from {role}'s round_{round_id}: {e}")
                continue
            choice_counts[final_choice] = choice_counts.get(final_choice, 0) + 1

        for choice, count in choice_counts.items():
            if count >= majority_count:
                logger.info(f"Consensus found on choice '{choice}' with {count}/{self.n_specialists} specialists.")
                return choice
            
        logger.info("No consensus found.")
        return None
    
    async def _aggregate(self):
        specialists_chat_history = dict(self.status_dict)
        for field_to_remove in ["label", "hadm_id", "problem"]:
            specialists_chat_history.pop(field_to_remove, None)

        specialists_str = json.dumps(specialists_chat_history, indent=4)

        user_prompt = (
            "No consensus has been reached among the specialists.\n"
            "You now have access to the entire conversation histories for all specialists.\n"
            "Please analyze each specialist's reasoning and final choice, then provide a single, definitive answer.\n"
            "Your answer should be the one best supported by their collective reasoning.\n\n"
            "Here is the complete conversation history for each specialist:\n\n"
            f"{specialists_str}\n\n"
            "After reviewing this material, please provide:\n"
            "1) A concise summary of the reasoning behind your final decision\n"
            "2) A single recommended choice: 'Yes' or 'No'\n"
            f"(indicating whether the patient has {self.problem})."
        )

        class AggregatedResponse(BaseModel):
            aggregated_reasoning: str = Field(..., description="Detailed reasoning behind the final choice.")
            aggregated_choice: Literal["Yes", "No"] = Field(
                ..., description=f"Single recommended choice for whether the patient has {self.problem}."
            )
        
        response = await self.llm_call(user_prompt, guided_={"guided_json": AggregatedResponse.model_json_schema()})

        data = safe_json_load(response)
        if not data:
            logger.error(f"Failed to parse aggregated response: {response}")
            return None
        return data
    
    async def run(self):
        while self.assignment_attempts < self.max_assignment_attempts:
            logger.info(f"Assignment attempt #{self.assignment_attempts + 1} started.")
            self.status_dict = await self._assign_specialists()
            
            panel = []
            for role in self.status_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"].keys():
                panel.append(
                    DynamicSpecialist(role, self.status_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"][role])
                )
                
            # Specialists analyze the note
            analyze_tasks = [asyncio.create_task(specialist.analyze_note(self.note, self.problem)) for specialist in panel]
            analyze_results = await asyncio.gather(*analyze_tasks)

            if any(r is None for r in analyze_results):
                logger.error("At least one specialist failed analysis; skipping this panel.")
                continue

            self.consensus_attempts += 1
            consensus_choice = self._check_consensus(self.assignment_attempts, self.consensus_attempts)
            if consensus_choice:
                self.status_dict["final"] = {
                    "final_choice": consensus_choice, 
                    "final_reasoning": "Consensus reached"
                }
                return self.status_dict
            
            # Attempt debate if no initial consensus
            while self.consensus_attempts < self.max_consensus_attempts:
                debate_tasks = [
                    asyncio.create_task(
                        specialist.debate(self.status_dict[f"panel_{self.assignment_attempts}"]["Collected Specialists"])
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
                    self.status_dict["final"] = {
                        "final_choice": consensus_choice, 
                        "final_reasoning": "Consensus reached"
                    }
                    return self.status_dict
                
            logger.info("No consensus reached after maximum consensus attempts among the panel.")
          
        logger.info("No consensus reached after maximum assignment attempts. Proceeding to aggregation.")
        aggregated_response = await self._aggregate()
        self.status_dict["final"] = {
            "final_choice": aggregated_response.get("aggregated_choice", "Error in aggregation."),
            "final_reasoning": aggregated_response.get("aggregated_reasoning", "Unable to parse aggregator response.")
        }
        return self.status_dict


class DynamicSpecialist(LLMAgent):
    def __init__(self, specialist: str, status: dict): 
        self.specialist = specialist
        self.expertise = status["expertise"]
        self.answer_history = status["answer_history"]
        self.round_id = 0
        system_prompt = (
            f"You are a {self.specialist}. "
            f"Your areas of expertise include:\n{self.expertise}\n"
            "Please analyze the patient's condition from the viewpoint of your specialty."
        )
        super().__init__(system_prompt)

    async def analyze_note(self, note: str, problem: str):
        self.round_id += 1

        class Response(BaseModel):
            reasoning: str = Field(..., description="Step-by-step reasoning leading to the final choice.")
            choice: Literal["Yes", "No"] = Field(..., description=f"Final choice indicating whether the patient has {problem}.")
        self.schema = Response.model_json_schema()

        user_prompt = (
            "Below are the Subjective (S) and Objective (O) sections from a SOAP note:\n\n"
            f"<SOAP>\n{note}\n</SOAP>\n\n"
            f"We need to determine if the patient has the following problem:\n\n<Problem>\n{problem}\n</Problem>\n\n"
            f"As a {self.specialist}, please provide:\n"
            "1) Your step-by-step reasoning\n"
            "2) A clear final answer: 'Yes' or 'No'\n"
        )

        try:
            response = await self.llm_call(user_prompt, guided_={"guided_json": self.schema})
        except Exception as e:
            logger.error(f"[{self.specialist}] analyze_note() LLM call failed: {e}")
            return None

        self.append_message(content=response)
        parsed_response = safe_json_load(response)
        if parsed_response:
            self.answer_history[f"round_{self.round_id}"] = parsed_response
            return parsed_response
        else:
            logger.error(f"[{self.specialist}] Failed to parse analysis response: {response}")
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
            "You now have additional information from other specialists. Here are their choices and reasoning:\n\n"
            f"{formatted_other_specialists}\n\n"
            "Please critique the reasoning of these other specialists and refine your own response if needed.\n"
            "You may change your final choice or keep your original conclusion.\n"
        )

        try:
            response = await self.llm_call(user_prompt, guided_={"guided_json": self.schema})
        except Exception as e:
            logger.error(f"[{self.specialist}] debate() LLM call failed: {e}")
            return None

        self.append_message(content=response)
        parsed_response = safe_json_load(response)
        if parsed_response:
            self.answer_history[f"round_{self.round_id}"] = parsed_response
            return parsed_response
        else:
            logger.error(f"[{self.specialist}] Failed to parse debate response: {response}")
            return None


async def main():
    df_path = "/home/yl3427/cylab/SOAP_MA/Input/filtered_merged_QA.csv"
    df = pd.read_csv(df_path, lineterminator='\n')
    logger.info("Loaded dataframe with %d rows.", len(df))

    results = []
    for idx, row in df.iterrows():
        logger.info(f"Processing row index {idx}")

        note_text = str(row["Subjective"]) + "\n" + str(row['Objective'])
        hadm_id = row["File ID"]
        problem = "sepsis"
        label = row["terms"]

        manager = Manager(
            note=note_text,
            hadm_id=hadm_id,
            problem=problem,
            label=label,
            n_specialists="auto",  
            consensus_threshold=0.8,
            max_consensus_attempts=3,
            max_assignment_attempts=2,
            static_specialists=None,    
            summarizer=None
        )

        # Run the manager's workflow
        result = await manager.run()
        print("Final Manager Result:")
        print(result)
        results.append(result)
    
    # Save results to a JSON file
    output_path = "/home/yl3427/cylab/SOAP_MA/Output/SOAP/manager_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to %s", output_path)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler()  # Print to console; you can also add FileHandler
        ]
    )
    
    asyncio.run(main())
