from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import re
from typing import Optional, Union, List, get_origin, get_args, Any, Dict, Literal
import inspect
# from __future__ import annotations
import asyncio
import json
import logging
import pandas as pd
from pydantic import BaseModel, Field, create_model
import math
import demjson3

logger = logging.getLogger(__name__)

def safe_json_load(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        logger.error("Standard json.loads failed: %s", e)
        try:
            logger.info("Attempting to parse with demjson3 as fallback.")
            result = demjson3.decode(s)
            logger.info("demjson3 successfully parsed the JSON.")
            return result
        except Exception as e2:
            logger.error("Fallback parsing with demjson3 also failed: %s. Returning original input.", e2)
            logger.error("Original input: %s", s)
            return s


class LLMAgent:
    def __init__(self, system_prompt: str, 
                 client=AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")):
        self.client = client
        self.messages = [{"role": "system", "content": system_prompt}]

    async def llm_call(self, user_prompt: str,
                       guided_: dict = None,
                       tools_descript: List[dict] = None, available_tools: Dict = None) -> Any:
        logger.debug(f"LLMAgent.llm_call() - user_prompt[:60]: {user_prompt[:60]}...")
        self.messages.append({"role": "user", "content": user_prompt})
        params = {
            # "model": "meta-llama/Llama-3.3-70B-Instruct",
            "model": '/secure_net/hf_model_cache/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/20b2ed1c4e9af44b9ad125f79f713301e27737e2',
            "messages": self.messages,
            "temperature": 0.3,
        }
        if guided_:
            logger.debug(f"Guided JSON/choice detected: {guided_}")
            params["extra_body"] = guided_

        if tools_descript:
            params["tools"] = tools_descript
            assert available_tools is not None, "available_tools must be provided if tools_descript is used."
            assert "provide_final_prediction" in available_tools, "provide_final_prediction tool is required."

        response = await self.client.chat.completions.create(**params)

        iter_count = 0
        if tools_descript:
            while True:
                if response.choices[0].message.tool_calls:
                    logger.info("Tool calls detected in response.")
                    self.messages.append({"role": "assistant", "tool_calls": response.choices[0].message.tool_calls})

                    for call in response.choices[0].message.tool_calls:
                        logger.debug(f"Tool call: {call}")
                        if call.function.name == "provide_final_prediction":
                            return call.function.arguments
                        
                        args = safe_json_load(call.function.arguments)
                        result = available_tools[call.function.name](**args)
                        self.messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        })
                else:
                    self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    self.messages.append({"role": "system", "content": "WARNING! Remember to call a tool."})
                    iter_count += 1
                    if iter_count > 3:
                        logger.error("Exceeded maximum iterations for tool calls.")
                        break
                    
                response = await self.client.chat.completions.create(**params)

        return response.choices[0].message.content
    
    def append_message(self, content, role='assistant'):
        logger.debug(f"Appending message with role='{role}' to conversation.")
        self.messages.append({"role": role, "content": content})
        return


class InitializerAgent(LLMAgent):
    def __init__(self, n_specialists: int):
        self.n_specialists = n_specialists
        system_prompt = (
            "You are an initializer agent in a multi-agent AI system designed to handle medical questions.\n"
            f"Your job is to select {self.n_specialists} medical specialists whose expertise best matches the user's query.\n"
            "For each specialist, specify their role and a list of relevant expertise areas related to the query.\n"
        )
        super().__init__(system_prompt)

    async def identify_specialists(self, query: str):
        logger.info("InitializerAgent: Identifying specialists.")
        class Specialist(BaseModel):
            specialist: str = Field(..., description="Role of the specialist")
            expertise: List[str] = Field(..., description="Areas of expertise for the specialist.")
        panel_dict = {f"Specialist_{i+1}": (Specialist, ...) for i in range(self.n_specialists)}
        SpecialistPanel = create_model("SpecialistPanel", **panel_dict)

        user_prompt = (
            "Here is the user's query:\n\n"
            f"<Query>\n{query}\n</Query>\n\n"
            "Based on the above query, identify the most suitable specialists."
        )
        response = await self.llm_call(user_prompt, guided_={"guided_json": SpecialistPanel.model_json_schema()})
        self.append_message(content=response)
        logger.debug(f"InitializerAgent response: {response}")
        return safe_json_load(response)


class SpecialistAgent(LLMAgent):
    def __init__(self, specialist: str, expertise: List[str]):
        self.specialist = specialist
        self.expertise = expertise
        system_prompt = (
            f"You are a {specialist}.\n"
            f"Your expertise includes:\n{expertise}\n"
            f"Analyze the user's query from the perspective of a {specialist}."
        )
        super().__init__(system_prompt)

    async def analyze_query(self, query: str, choices: List[str]):
        logger.info(f"[{self.specialist}] Analyzing query...")
        self.query = query
        self.choices = tuple(choices)
        choices_str = ', '.join(choices)

        user_prompt = (
            "Here is the query of interest:\n\n"
            f"<Query>\n{query}\n</Query>\n\n"
            f"The possible answers are: {choices_str}.\n"
            f"From your perspective as a {self.specialist}, first provide step-by-step reasoning (rationale), "
            "and then clearly state your final answer.\n\n"
        )

        class Response(BaseModel):
            reasoning: str = Field(..., description="Step-by-step reasoning leading to the final choice")
            choice: Literal[self.choices] = Field(..., description="Final choice")

        response = await self.llm_call(user_prompt, guided_={"guided_json": Response.model_json_schema()})
        self.append_message(content=response)
        logger.debug(f"[{self.specialist}] analyze_query response: {response}")
        return safe_json_load(response)
    
    async def debate(self, agents: Dict[str, Any]):
        logger.info(f"[{self.specialist}] Debating with other specialists.")
        other_specialists = {}
        for name, value in agents.items():
            if name != self.specialist:
                other_specialists[name] = value

        formatted_other_specialists = json.dumps(other_specialists, indent=4)
        user_prompt = (
            "Regarding the previous query, other specialists have also provided their reasoning and choices.\n"
            "Critically evaluate the reasoning and choice of those specialists.\n\n"
            f"Specialists and their choices:\n{formatted_other_specialists}\n\n"
            "Considering the newly provided perspectives, refine your own reasoning and choice.\n"
            "You can change your choice or stick with the original one.\n\n"
        )

        class Response(BaseModel):
            reasoning: str = Field(..., description="Step-by-step reasoning leading to final choice")
            choice: Literal[self.choices] = Field(..., description="Final choice")

        response = await self.llm_call(user_prompt, guided_={"guided_json": Response.model_json_schema()})
        self.append_message(content=response)
        logger.debug(f"[{self.specialist}] debate response: {response}")
        return safe_json_load(response)


class AggregatorAgent(LLMAgent):
    def __init__(self):
        system_prompt = (
            "You are the aggregator agent in a multi-agent AI system for medical queries.\n"
            "You have access to each specialist's entire chat history.\n"
            "Your job is to read those full conversations, analyze their reasoning and any conflicts, "
            "and then provide a single, definitive answer to the user.\n"
            "Provide a clear explanation for your final conclusion."
        )
        super().__init__(system_prompt)

    async def aggregate(self, query: str, choices: List[str], specialists_chat_history: Dict[str, Any]):
        logger.info("AggregatorAgent: Aggregating final answer from all specialists' chat history.")
        specialists_str = json.dumps(specialists_chat_history, indent=4)

        user_prompt = (
            f"Here is the query of interest:\n\n"
            f"<Query>\n{query}\n</Query>\n\n"
            "Below is the *entire conversation history* for each specialist:\n\n"
            f"{specialists_str}\n\n"
            "Please review all these conversations in detail and produce one single, definitive final answer. "
            "If there is no unanimous or majority choice, choose the answer best supported by the specialists' reasoning. "
            "Clearly justify your reasoning, then provide your final recommended answer."
        )

        class AggregatedResponse(BaseModel):
            aggregated_reasoning: str = Field(..., description="Detailed reasoning behind final choice")
            aggregated_choice: Literal[tuple(choices)] = Field(..., description="Single recommended choice")

        response = await self.llm_call(user_prompt, guided_={"guided_json": AggregatedResponse.model_json_schema()})
        self.append_message(content=response)
        logger.debug(f"AggregatorAgent response: {response}")
        return safe_json_load(response)


def check_consensus(status_dict: Dict[str, Any]) -> str:
    """
    Returns the consensus choice if >= 80% of specialists agree, else returns None.
    """
    logger.info("Checking for consensus among specialists.")
    specialists_count = len(status_dict)
    consensus_threshold = math.ceil(0.8 * specialists_count)

    choice_counts = {}
    for _, specialist_data in status_dict.items():
        final_choice = specialist_data['response_after_debate']['choice']
        choice_counts[final_choice] = choice_counts.get(final_choice, 0) + 1

    for choice, count in choice_counts.items():
        if count >= consensus_threshold:
            logger.info(f"Consensus found on choice '{choice}' with {count}/{specialists_count} specialists.")
            return choice
    logger.info("No consensus found.")
    return None

############################################################################################################
# --------------------------------
# 3) PROCESS A SINGLE ROW/QUERY
# --------------------------------
async def process_single_query(
    question_text: str,
    ground_truth: str,
    choices: List[str],
    n_specialists: int) -> Dict[str, Any]:
    """
    Given a single query (question + ground_truth + multiple choices), 
    run the multi-agent system (Initializer -> Specialists -> Debates -> Aggregator if needed).
    Return the final dictionary containing all the specialists' output and aggregator results.
    """

    # 1. Initialize specialists
    initializer = InitializerAgent(n_specialists=n_specialists)
    json_resp = await initializer.identify_specialists(query=question_text)
    if not isinstance(json_resp, dict):
        logger.error("Invalid JSON output from initializer; skipping this query.")
        return {}  # Skip processing and continue to the next query

    # Build specialists status dict
    specialists_status = {}
    for _, agent_info in json_resp.items():
        specialist_name = agent_info["specialist"]
        expertise = agent_info["expertise"]
        specialists_status[specialist_name] = {"expertise": expertise}
    
    # 2. Run analyze_query for each specialist in parallel
    async def analyze_specialist(specialist_name: str, status: Dict[str, Any], query: str, choices: List[str]):
        specialist_agent = SpecialistAgent(specialist=specialist_name, expertise=status["expertise"])
        status["instance"] = specialist_agent
        message = await specialist_agent.analyze_query(query=query, choices=choices)
        if not isinstance(message, dict):
            logger.error(f"[{specialist_name}] Invalid JSON output from specialist; skipping this specialist.")
            return None
        status["original_response"] = message
        logger.info(f"[{specialist_name}] Completed analyze_query.")
        return specialist_name

    analyze_tasks = [
        asyncio.create_task(analyze_specialist(name, status, question_text, choices))
        for name, status in specialists_status.items()
    ]
    analyze_results = await asyncio.gather(*analyze_tasks)
    if any(r is None for r in analyze_results):
        logger.error("At least one specialist failed; skipping this query.")
        return {}  # Skip processing and continue to the next query

    # Build a minimal dictionary for debate (remove 'instance')
    input_specialists_dict = {
        specialist_name: {
            k: v for k, v in specialist_data.items() 
            if k != "instance"
        }
        for specialist_name, specialist_data in specialists_status.items()
    }

    # 3. Debate step, also in parallel
    async def debate_specialist(specialist_name: str, status: Dict[str, Any], specialists_dict: Dict[str, Any]):
        specialist_agent = status["instance"]
        message = await specialist_agent.debate(specialists_dict)
        if not isinstance(message, dict):
            logger.error(f"[{specialist_name}] Invalid JSON output during debate; skipping this specialist.")
            return None
        status["response_after_debate"] = message
        specialists_dict[specialist_name]["response_after_debate"] = message
        logger.info(f"[{specialist_name}] Completed debate.")
        return specialist_name

    debate_tasks = [
        asyncio.create_task(debate_specialist(name, status, input_specialists_dict))
        for name, status in specialists_status.items()
    ]
    debate_results = await asyncio.gather(*debate_tasks)
    if any(r is None for r in debate_results):
        logger.error("At least one specialist failed during debate; skipping this query.")
        return {}  # Skip processing and continue to the next query

    # 4. Check consensus
    consensus_choice = check_consensus(input_specialists_dict)
    aggregator_result = None

    if consensus_choice is not None:
        logger.info(f"Consensus reached: {consensus_choice}")
        input_specialists_dict["Aggregator"] = {
            "final_choice": consensus_choice, 
            "final_reasoning": "Consensus reached"
        }
    else:
        logger.info("No consensus reached; enabling aggregator path...")
        aggregator = AggregatorAgent()
        aggregated_response = await aggregator.aggregate(
            query=question_text,
            choices=choices,
            specialists_chat_history=input_specialists_dict
        )
        if not isinstance(aggregated_response, dict):
            logger.error("Invalid JSON output from aggregator; skipping this query.")
            return {}  # Skip processing and continue to the next query
        
        final_choice = aggregated_response['aggregated_choice']
        final_reasoning = aggregated_response['aggregated_reasoning']

        logger.info(f"Aggregator final choice: {final_choice}")
        logger.info(f"Aggregator reasoning: {final_reasoning}")

        aggregator_result = {
            "final_choice": final_choice,
            "final_reasoning": final_reasoning
        }
        input_specialists_dict["Aggregator"] = aggregator_result

    # Add question and ground_truth for reference
    input_specialists_dict["Question"] = question_text
    input_specialists_dict["Answer"] = ground_truth

    return input_specialists_dict


async def process_multiple_queries(
    qa_df: pd.DataFrame,
    choices: List[str],
    n_specialists: int,
    max_concurrency: int = 5
) -> List[Dict[str, Any]]:
    """
    Process multiple rows (queries) in `qa_df` asynchronously.
    Each row is passed to `process_single_query`.
    
    :param qa_df: DataFrame with columns ["question", "choice", "ground_truth"] at least.
    :param choices: A list of all possible answer choices, e.g. ["A", "B", "C", "D", "E"].
    :param n_specialists: Number of specialists to initialize for each query.
    :param max_concurrency: Limit on how many queries to process simultaneously.
    :return: A list of result dictionaries, one per row in `qa_df`.
    """

    # This semaphore keeps at most `max_concurrency` tasks running at once
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_single_query(row_idx: int, row: pd.Series):
        """
        This inner function is used to call `process_single_query` with concurrency control.
        """
        async with semaphore:
            logger.info(f"Starting row {row_idx}")
            question_text = row["question"] + "\n" + str(row["choice"])
            ground_truth = str(row["ground_truth"])
            result = await process_single_query(
                question_text=question_text,
                ground_truth=ground_truth,
                choices=choices,
                n_specialists=n_specialists
            )
            logger.info(f"Finished row {row_idx}")
            return result

    tasks = [
        asyncio.create_task(run_single_query(i, row))
        for i, row in qa_df.iterrows()
    ]

    # Wait for all tasks to complete
    all_results = await asyncio.gather(*tasks)

    # `all_results` is a list of return values from each `run_single_query`
    return all_results

async def main():

    logger.info("===== MAIN START =====")

    # Example CSV loading
    # df_path = "/home/yl3427/cylab/SOAP_MA/Input/step1_ALL.csv"
    df_path = "/home/yl3427/cylab/SOAP_MA/Input/filtered_merged_QA.csv"
    qa_df = pd.read_csv(df_path, lineterminator='\n')  # columns: idx, question, choice, ground_truth, qn_num

    # qa_df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/Input/SOAP_5_problems.csv')
    logger.info("Loaded dataframe with %d rows.", len(qa_df))


    ################# 'process_single_query' Example usage #################
    results = []
    for idx, row in qa_df.iterrows():
        # if row["qn_num"] not in [13, 42]:
        # if row["File ID"] not in ['123147.txt']:
        #     continue

        logger.info(f"Processing row index {idx}")

        question_text = row["question"] + "\n" + str(row["choice"])
        ground_truth = str(row["ground_truth"])
        # patient_info = str(row["Subjective"]) + "\n" + str(row['Objective'])
        # question_text = f"""
        # Based on the following patient report, does the patient have congestive heart failure?"

        # {patient_info}
        # """
        # ground_truth = str(row["terms"])
        

        # Run the multi-agent system for this single query
        result_dict = await process_single_query(
            question_text=question_text,
            ground_truth=ground_truth,
            choices=["A", "B", "C", "D", "E"],
            # choices=["Yes", "No"],
            n_specialists=5
        )
        # result_dict["File ID"] = row["File ID"]
        result_dict["qn_num"] = row["qn_num"]

        # Store result for later evaluation
        results.append(result_dict)

        if idx % 10 == 0:
            output_json_path = f"/home/yl3427/cylab/SOAP_MA/Output/MedicalQA/merged_{idx}_mistral.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved aggregated results to {output_json_path}")

    # OPTIONAL: Save results to JSON
    output_json_path = "/home/yl3427/cylab/SOAP_MA/Output/MedicalQA/merged_final_mistral.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved aggregated results to {output_json_path}")



    # ################ 'process_multiple_queries' Example usage #################
    # results = await process_multiple_queries(
    #     qa_df=qa_df,
    #     choices=["A", "B", "C", "D", "E"],
    #     n_specialists=3,
    #     max_concurrency=5  # you can tune this
    # )

    # # Optionally save the results to JSON
    # output_path = "aggregator_results.json"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    # logger.info(f"Saved {len(results)} aggregator results to {output_path}")

    logger.info("===== MAIN END =====")

# # If you're in a notebook, just do:
# await main()

# If you're in a script, you can do:
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
            logging.FileHandler('log/0327_MA_MedicalQA_mistral_merged.log', mode='a'),  # Write to file
            logging.StreamHandler()                     # Print to console
        ]
    )

    asyncio.run(main())