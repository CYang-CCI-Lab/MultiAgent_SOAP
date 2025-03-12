import asyncio
import json
import logging
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field, create_model
from openai import AsyncOpenAI
import demjson3

# Set up logging
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
            return s

class BaselineLLMAgent:

    def __init__(
        self,
        system_prompt: str,
        client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy"),
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    ):
        self.client = client
        self.model_name = model_name
        self.messages = [{"role": "system", "content": system_prompt}]

    async def ask(self, user_prompt: str, guided_schema: Dict[str, Any] = None) -> Any:
        """
        Sends a single query to the LLM and returns the raw response or JSON (if `guided_schema` is provided).
        """
        self.messages.append({"role": "user", "content": user_prompt})
        params = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": 0.5,
        }
        if guided_schema is not None:
            # e.g. pydantic style schema instructions
            params["extra_body"] = guided_schema

        response = await self.client.chat.completions.create(**params)
        content = response.choices[0].message.content
        return content

#
# A convenience function that runs the "baseline" query for one question.
#
async def process_single_query_baseline(
    question_text: str,
    choices: List[str]
) -> Dict[str, Any]:

    system_prompt = "You are a medical question-answering system."

    # 2) We use a pydantic model to “guide” the LLM to produce a structured response.
    class BaselineResponse(BaseModel):
        reasoning: str = Field(..., description="Step-by-step reasoning leading to the final choice")
        choice: Literal[tuple(choices)] = Field(..., description="Final choice")

    # 3) Build the user prompt:
    choices_str = ", ".join(choices)

    user_prompt = (
        "Here is the query:\n\n"
        f"<Query>\n{question_text}\n</Query>\n\n"
        f"The possible answers are: {choices_str}.\n"
        f"First provide step-by-step reasoning (rationale), "
        "and then clearly state your final answer.\n\n"
    )

    # 4) Initialize the baseline LLM agent
    agent = BaselineLLMAgent(system_prompt=system_prompt)

    # 5) Call the LLM once and parse JSON
    raw_output = await agent.ask(
        user_prompt,
        guided_schema={"guided_json": BaselineResponse.model_json_schema()}
    )
    parsed = safe_json_load(raw_output)
    
    # 6) If parsing fails or is not properly formatted, you can do fallback
    if not isinstance(parsed, dict):
        logger.warning("LLM response is not valid JSON or missing expected fields. Returning raw text.")
        final_choice = "Unknown"
        rationale = str(raw_output)
    else:
        final_choice = parsed["choice"]
        rationale = parsed["reasoning"]

    # 7) Return a dictionary that you can compare to the multi-agent approach
    return {
        "BaselineRationale": rationale,
        "BaselineChoice": final_choice
    }

sem = asyncio.Semaphore(10)
async def process_with_semaphore(question_text, choices):
    async with sem:
        return await process_single_query_baseline(question_text, choices)
    
async def process_multiple_queries_baseline(
    qa_data: List[Dict[str, Any]],
    choices: List[str]
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

        tasks.append(process_with_semaphore(question_text, choices))
    
    all_results = await asyncio.gather(*tasks)

    for idx, result in zip(processed_idx, all_results):
        qa_data[idx].update(result)
        
    return qa_data


async def main():
    # Suppose you have data like:
    input_json_path = "/home/yl3427/cylab/SOAP_MA/Output/MedicalQA/step2_final.json"
    with open(input_json_path, "r") as f:
        sample_data = json.load(f)
    # sample_data = [
    #     {"Question": "What is the best initial therapy for pneumonia?\nA) Antibiotics\nB) Surgery\nC) Radiation\nD) Physical therapy\nE) Do nothing", 
    #      "ground_truth": "A"},
    #     {"Question": "A patient with a headache might have:\nA) Migraine\nB) Stubbed toe\nC) Carpal tunnel\nD) Cirrhosis\nE) Diabetes",
    #      "ground_truth": "A"},
    # ]

    # The multiple-choice labels might be the same for all or differ per question:
    choice_labels = ["A", "B", "C", "D", "E"]

    new_data = await process_multiple_queries_baseline(sample_data, choice_labels)
    with open(f"{input_json_path.split('.')[0]}_with_baseline.json", "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    logger.info("Finished processing all items.")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
            logging.FileHandler('0311_Baseline_MedicalQA_step2_async.log', mode='w'),  # Write to file
            logging.StreamHandler()                     # Print to console
        ]
    )

    asyncio.run(main())
