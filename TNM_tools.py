from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import re
from typing import Optional, Union, List, get_origin, get_args, Any, Dict, Literal
import inspect
import asyncio
import json
import logging
import pandas as pd
from pydantic import BaseModel, Field, create_model
import math
import demjson3
logger = logging.getLogger(__name__)
def safe_json_load(s: str) -> Any:
    """
    Attempts to parse a JSON string using the standard json.loads.
    If that fails (e.g. due to an unterminated string), it will try using
    a more forgiving parser (demjson3). If both attempts fail,
    the original string is returned.
    """
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
            return s

def generate_tools_spec(*functions):
    # Mapping of Python types to JSON Schema types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null"
    }
    tools = []
    
    for func in functions:
        func_name = func.__name__
        func_description = (func.__doc__ or "").strip()
        sig = inspect.signature(func)
        
        properties = {}
        required = []
        
        for param in sig.parameters.values():
            # Skip *args and **kwargs as they cannot be described in JSON schema easily
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            
            param_name = param.name
            annotation = param.annotation
            json_type = "string"  # default type for fallback

            if annotation is not inspect._empty:
                origin = get_origin(annotation)
                
                # Handle Literal types (e.g., Literal["T", "N", "M"])
                if origin is Literal:
                    literal_args = get_args(annotation)
                    
                    # If all literal args are strings, produce a string enum
                    if all(isinstance(arg, str) for arg in literal_args):
                        properties[param_name] = {
                            "type": "string",
                            "enum": list(literal_args)
                        }
                    # If all are integers, produce an integer enum, etc.
                    elif all(isinstance(arg, int) for arg in literal_args):
                        properties[param_name] = {
                            "type": "integer",
                            "enum": list(literal_args)
                        }
                    else:
                        # Fallback if the Literal contains mixed or unsupported types
                        properties[param_name] = {"type": "string"}
                
                # Handle Optional[X] or Union[X, None]
                elif origin is Union:
                    union_args = [t for t in get_args(annotation) if t is not type(None)]
                    if len(union_args) == 1:
                        # e.g. Optional[str] -> just str
                        real_type = union_args[0]
                        origin2 = get_origin(real_type)
                        
                        if origin2 is Literal:
                            # If inside an Optional[Literal[...]]
                            literal_args = get_args(real_type)
                            if all(isinstance(arg, str) for arg in literal_args):
                                properties[param_name] = {
                                    "type": "string",
                                    "enum": list(literal_args)
                                }
                            elif all(isinstance(arg, int) for arg in literal_args):
                                properties[param_name] = {
                                    "type": "integer",
                                    "enum": list(literal_args)
                                }
                            else:
                                properties[param_name] = {"type": "string"}
                        else:
                            # Map direct type to JSON schema
                            json_type = type_map.get(real_type, "string")
                            properties[param_name] = {"type": json_type}
                    else:
                        # More complex Unions not automatically handled; fallback to string
                        properties[param_name] = {"type": "string"}
                
                # If it's a known type (str, int, etc.)
                elif annotation in type_map:
                    json_type = type_map[annotation]
                    properties[param_name] = {"type": json_type}
                
                # Handle typing.List[...] or typing.Dict[...] 
                elif origin in type_map:
                    json_type = type_map[origin]
                    if json_type == "array":
                        # For list[...] or array
                        item_type = "string"
                        args = get_args(annotation)
                        if args and args[0] in type_map:
                            item_type = type_map[args[0]]
                        properties[param_name] = {
                            "type": "array",
                            "items": {"type": item_type}
                        }
                    elif json_type == "object":
                        # For dict[...] or any unhandled complex mapping
                        properties[param_name] = {"type": "object"}
                
                else:
                    # Fallback if we can't detect the type
                    properties[param_name] = {"type": "string"}
            
            else:
                # No annotation; assume string
                properties[param_name] = {"type": "string"}

            # Mark as required if no default value
            if param.default is inspect._empty:
                required.append(param_name)
        
        tool_dict = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": func_description,
                "parameters": {
                    "type": "object",
                    "properties": properties
                }
            }
        }
        if required:
            tool_dict["function"]["parameters"]["required"] = required
        
        tools.append(tool_dict)
    
    return tools

def get_t_stage_rule(stage: Literal["T1", "T2", "T3", "T4"]) -> str:
    """
    Returns the rule for the given pathological T stage of breast cancer
    according to the AJCC Cancer Staging Manual (7th edition).

    Inputs:
        stage (str): The T stage ('T1', 'T2', 'T3', 'T4').

    Returns:
        str: The rule corresponding to the T stage.
    """
    stage_rules = {
        'T1': 'The tumor is 2 centimeters (cm) or less in greatest dimension.',
        'T2': 'The tumor is more than 2 cm but not more than 5 cm in greatest dimension.',
        'T3': 'The tumor is more than 5 cm in greatest dimension.',
        'T4': 'The tumor is of any size with direct extension to the chest wall and/or to the skin (ulceration or skin nodules).'
    }
    return stage_rules.get(stage.upper(), 'Invalid T stage. Please enter T1, T2, T3, or T4.')

def extract_information(info_to_extract: Union[List[str], str]) -> Dict[str, str]:
    """
    Extracts relevant information from a given pathology text.

    Inputs:
        info_to_extract (List[str]): A list of information fields to be extracted,
            e.g. ["tumor_size", "depth_of_invasion", ...].
    
    Returns:
        Dict[str, str]: A dictionary mapping each requested field to the extracted information.
    """
    if isinstance(info_to_extract, str):
        info_to_extract = [info_to_extract]

def compare_numerical_values(
    value: float,
    min_value: float = None,
    max_value: float = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True
) -> bool:
    """
    Compares a given numeric value against optional minimum and maximum thresholds.
    Args:
        value (float): The numeric value to compare.
        min_value (float, optional): The lower threshold. 
                                     If None, no lower bound check is performed.
        max_value (float, optional): The upper threshold. 
                                     If None, no upper bound check is performed.
        inclusive_min (bool): Whether the comparison with min_value 
                              should be inclusive (value >= min_value) 
                              or exclusive (value > min_value).
        inclusive_max (bool): Whether the comparison with max_value 
                              should be inclusive (value <= max_value) 
                              or exclusive (value < max_value).

    Returns:
        bool: True if the value satisfies all specified boundary conditions; 
              False otherwise.

    Examples:
        compare_numerical_values(3.2, min_value=2, max_value=5) 
            -> True (assuming inclusive checks)
        compare_numerical_values(2, min_value=2, max_value=5, inclusive_min=False)
            -> False, since 2 is not strictly > 2
    """
    if min_value is not None:
        if inclusive_min:
            if value < min_value:
                return False
        else:
            if value <= min_value:
                return False

    if max_value is not None:
        if inclusive_max:
            if value > max_value:
                return False
        else:
            if value >= max_value:
                return False

    return True

def produce_final_staging_response(agent_response: str) -> Dict[str, str]:
    """
    Takes the agent's final response and reformat it into a JSON schema with 'reasoning' and 'stage' as keys.

    Args:
        agent_response (str): The final response from the agent (after all internal processing).

    Returns:
        Dict[str, str]: A dictionary containing two keys: 'reasoning' and 'stage'.
    """

    # Example: we create a second LLM client specifically for formatting the final output
    formatting_client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    formatting_model = "meta-llama/Llama-3.3-70B-Instruct"

    prompt = f"""You are given the final reasoning and conclusions about a cancer T-staging task:

\"\"\"{agent_response}\"\"\"

Please provide valid JSON (and ONLY JSON, without extra text) with the following structure:
{{
"reasoning": "A brief explanation of the reasoning that led to the stage conclusion.",
"stage": "The final T stage (e.g. T1, T2, T3...)"
}}

Make sure the output is strictly valid JSON.
"""
        
    class ResponseStage(BaseModel):
        reasoning: str = Field(
            description="A step-by-step explanation for how you arrived at the predicted T stage."
        )
        stage: Literal["T1", "T2", "T3", "T4"] = Field(
            description="The final predicted T stage (T1, T2, T3, or T4)."
        )


    response = formatting_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=formatting_model,
        temperature=0.1,
        extra_body={"guided_json": ResponseStage.model_json_schema()}
        )

    return response.choices[0].message.content


available_tools = {
    "get_t_stage_rule": get_t_stage_rule,
    "extract_information": extract_information,
    "compare_numerical_values": compare_numerical_values,
    "produce_final_staging_response": produce_final_staging_response
}
tools = generate_tools_spec(*available_tools.values())

class LLMAgent:

    def __init__(
        self,
        system_prompt: str,
        client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy"),
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    ):
        self.client = client
        self.model_name = model_name
        self.messages = [{"role": "system", "content": system_prompt}]

    async def monologue(self, query: str) -> Any:
        self.user_prompt = query
        user_prompt = f""""Here is the user's query:\n\n"
            f"<Query>\n{query}\n</Query>\n\n"""
        self.messages.append({"role": "user", "content": user_prompt})
        
        plan_instruct = """Plan your actions or think privately to address the user query. Then, provide a monologue with your inner thoughts and planned actions.
        This monolgue is not shared with the user, and is for your planning purposes only.
        Calling 'produce_final_staging_response' tool is the ONLY action that sends a response to the user and it should be done ONLY AFTER executing all planned steps.
        In other words, the last action(element) in your plan(tools_queued) should be 'produce_final_staging_response' to send the final response(your final staging prediction and reasoning) to the user.
        """

        self.messages.append({"role": "system", "content": plan_instruct})

        class MonologueResponse(BaseModel):
            inner_monologue: str = Field(..., description="Inner monologue to plan actions or think privately")
            tools_queued: List[str] = Field(..., description="List of tools to be used in the next step")

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=0.5,
            tools=tools,
            extra_body={"guided_json": MonologueResponse.model_json_schema()}
        )

        logger.debug(f"Monologue response: {response}")


        max_retry = 3
        while response.choices[0].message.tool_calls:
            logging.error("Monologue step should not contain tool calls. Only plan your actions.")
            system_warning = "Do not call tools yet in this monologue step. Only plan your actions."
            self.messages.append({"role": "system", "content": system_warning})
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=0.5,
                tools=tools,
                extra_body={"guided_json": MonologueResponse.model_json_schema()}
            )
            max_retry -= 1
            if max_retry == 0:
                logger.error("Exceeded maximum retry attempts. Skipping this report.")
                return None

        content = response.choices[0].message.content
        logger.debug(f"Monologue final content: {content}")
        self.messages.append({"role": "assistant", "content": content})
        return content
    
    async def execute_plan(self, tools_queued: List[str]):
        for idx in range(len(tools_queued)):
            if idx == len(tools_queued) - 1:
                system_prompt = "You have reached the final step. It's time to call the 'produce_final_staging_response' tool to generate the final response."
            else:
                system_prompt = f"""Among the available tools, you have chosen to use the '{tools_queued[idx]}' tool for the {idx+1}-th step.
            Please call this tool with the appropriate parameters to gather the necessary information for the cancer T-staging task."""
            
            self.messages.append({"role": "system", "content": system_prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=0.5,
                tools=tools
            )
            logger.debug(f"execute_plan response: {response}")

            max_retry = 3
            while not response.choices[0].message.tool_calls:
                system_warning = "You must call a tool in this step. Please use the selected tool."
                self.messages.append({"role": "system", "content": system_warning})
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    temperature=0.5,
                    tools=tools
                )
                max_retry -= 1
                if max_retry == 0:
                    logger.error("Exceeded maximum retry attempts. Skipping this report.")
                    return None
        
            self.messages.append({"role": "assistant", "tool_calls": response.choices[0].message.tool_calls})

            for call in response.choices[0].message.tool_calls:
                logger.debug(f"Tool call: {call}")             
                args = safe_json_load(call.function.arguments)
                if call.function.name == "extract_information":
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages= [{"role": "user", "content": "Extract information for {args} from the following pathology report.\n\n{report}".format(args=str(args), report=self.user_prompt.split("Choose from T1, T2, T3, T4.\n\n")[1])}],
                        temperature=0.1)
                    result = response.choices[0].message.content
                    logger.debug(f"Extract information result: {result}")
                else:
                    func = available_tools[call.function.name]
                    result = func(**args)
                    logger.debug(f"Tool result: {result}")

                self.messages.append({
                "role": "function",
                "content": json.dumps(result) if not isinstance(result, str) else result,
                "tool_call_id": call.id,
                "name": call.function.name,
                })

        return result # Assuming it is the dictionary-like string returned by 'produce_final_staging_response'

            


async def process_single_query_baseline(
    question_text: str
) -> Dict[str, Any]:

    system_prompt =  """You are a medical question-answering system focused on breast cancer T-staging tasks.

Core Objectives
1. Queue of Tasks: You have an internal queue of tasks (e.g., extracting information from pathology reports, comparing numeric values, applying T-staging rules). You must always clear these tasks before producing a final message for the user.
2. Tool Usage: Call the relevant tools (like extract_information, compare_numerical_values, etc.) to gather or analyze data.
3. Monologue vs. User Messages:
    - Your inner monologue is private. This is where you plan and reason silently. It is never revealed to the user.
    - Only the final output from produce_final_staging_response is visible to the user.
    - Do not send partial or intermediate reasoning to the user.

Multi-Step Thought Process
- Plan your approach in a short, private inner monologue.
- If your monologue requires you to gather data or perform staging checks, call the appropriate tools.
- Final Output: Only after you have finished using tools and have your conclusion do you call produce_final_staging_response. This produces the JSON output for the user containing your reasoning and the final T stage.

Style and Constraints
- Conciseness: Keep your inner monologue under 50 words and succinctly capture your plan.
- No Extraneous Details: Avoid discussing your internal structure, any mention of large language models, or your AI nature.
- Keep the Focus on T-Staging: You are a medical professional providing helpful explanations and staging outcomes based on the given pathology text.

Memory and Task Handling
- Before you respond publicly, ensure your tasks queue is completed.
- You may rely on your tool calls (extract_information, compare_numerical_values, get_t_stage_rule) to gather or confirm data.

By following these guidelines, you will provide a seamless, professional conversation flow for T-staging of breast cancer while handling all required sub-tasks behind the scenes."""

    agent = LLMAgent(system_prompt=system_prompt)

    raw_output = await agent.monologue(question_text)
    parsed = safe_json_load(raw_output)

    if not isinstance(parsed, dict):
        logger.warning("LLM response is not valid JSON or missing expected fields. Returning raw text.")
        final_choice = "Unknown"
        rationale = str(raw_output)
        return {
            "Rationale": rationale,
            "Choice": final_choice
        }
    else:
        inner_monologue = parsed["inner_monologue"]
        tools_queued = parsed["tools_queued"]

        raw_final_response = await agent.execute_plan(tools_queued)
        final_response = safe_json_load(raw_final_response)
        if not isinstance(final_response, dict):
            logger.warning("LLM response is not valid JSON or missing expected fields. Returning raw text.")
            final_choice = "Unknown"
            rationale = str(raw_final_response)
        else:
            final_choice = final_response["stage"]
            rationale = final_response["reasoning"]
        return {
            "Monologue": inner_monologue,
            "ToolsQueued": str(tools_queued),
            "Rationale": rationale,
            "Choice": final_choice
        }


async def main():

    sem = asyncio.Semaphore(10)
    async def process_with_semaphore(question_text):
        async with sem:
            return await process_single_query_baseline(question_text)
        

    async def process_multiple_queries_baseline(
        qa_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tasks = []

        processed_idx = []
        for idx, item in enumerate(qa_data):
            if "Question" not in item:
                logger.warning(f"Skipping item {idx} because it does not have a 'Question' field.")
                continue

            logger.info(f"Processing item {idx}...")
            processed_idx.append(idx)
            question_text = item["Question"]

            tasks.append(process_with_semaphore(question_text))
        
        all_results = await asyncio.gather(*tasks)

        for idx, result in zip(processed_idx, all_results):
            qa_data[idx].update(result)
            
        return qa_data


    #############
    with open('/home/yl3427/cylab/SOAP_MA/Output/TNM/brca_only_with_baseline.json', 'r') as file:
        sample_data = json.load(file)
    
    sample_data = sample_data[100:500]

    new_data = await process_multiple_queries_baseline(sample_data)
    
    unknown_indices = [i for i, entry in enumerate(new_data) if entry.get("Choice") == "Unknown"]
    if unknown_indices:
        logger.info(f"Found {len(unknown_indices)} items with 'Unknown' result. Re-running those...")
        unknown_tasks = []
        for idx in unknown_indices:
            question_text = new_data[idx]["Question"]
            unknown_tasks.append(process_with_semaphore(question_text))
        unknown_results = await asyncio.gather(*unknown_tasks)

        for idx, result in zip(unknown_indices, unknown_results):
            new_data[idx].update(result)



    output_json_path = "/home/yl3427/cylab/SOAP_MA/Output/TNM/brca_final_with_baseline_500.json"
    with open(output_json_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    logger.info("Finished processing all items.")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
            logging.FileHandler('log/0314_TNM_brca_async.log', mode='a'),  # Write to file
            logging.StreamHandler()                     # Print to console
        ]
    )

    asyncio.run(main())
