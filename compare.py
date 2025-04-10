import asyncio
import json
import logging
from typing import List, Dict, Any, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from utils import safe_json_load

logger = logging.getLogger(__name__)

# Reuse BaselineLLMAgent class
class BaselineLLMAgent:
    def __init__(
        self,
        system_prompt: str,
        client=AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy"),
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    ):
        self.client = client
        self.model_name = model_name
        self.messages = [{"role": "system", "content": system_prompt}]

    async def ask(self, user_prompt: str, guided_schema: Optional[Dict[str, Any]] = None) -> Optional[str]:
        self.messages.append({"role": "user", "content": user_prompt})
        params = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": 0.5
        }
        if guided_schema:
            params["extra_body"] = guided_schema

        try:
            response = await self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error while calling LLM: %s", e)
            return None

# Reuse process_single_query_baseline
async def process_single_query_baseline(note: str, problem: str) -> Optional[Dict[str, Any]]:
    system_prompt = "You are a medical question-answering system."

    class BaselineResponse(BaseModel):
        reasoning: str = Field(..., description="Step-by-step reasoning leading to the final choice.")
        choice: Literal["Yes", "No"] = Field(
            ...,
            description=f"Final choice indicating whether the patient has {problem}."
        )

    user_prompt = (
        "Here are the Subjective (S) and Objective (O) parts of a SOAP note:\n\n"
        f"<SOAP>\n{note}\n</SOAP>\n\n"
        "Based on this information, does the patient have the following problem?\n\n"
        f"<Problem>\n{problem}\n</Problem>\n\n"
        "Please provide:\n"
        "1. Your step-by-step reasoning.\n"
        "2. Your choice: 'Yes' or 'No'."
    )

    agent = BaselineLLMAgent(system_prompt=system_prompt)
    schema = {"guided_json": BaselineResponse.model_json_schema()}
    raw_output = await agent.ask(user_prompt, guided_schema=schema)

    parsed = safe_json_load(raw_output)
    if parsed:
        return {
            "BaselineRationale": parsed["reasoning"],
            "BaselineChoice": parsed["choice"]
        }
    else:
        logger.error("Failed to parse the response as JSON: %s", raw_output)
        return {"BaselineRationale": "Unknown", "BaselineChoice": "Unknown"}

# Reuse process_multiple_queries_baseline
async def process_multiple_queries_baseline(
    data_entries: List[Dict[str, Any]],
    concurrency: int = 10
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def process_with_semaphore(note_text: str, problem_text: str) -> Optional[Dict[str, Any]]:
        async with sem:
            return await process_single_query_baseline(note_text, problem_text)

    tasks = []
    processed_indices = []

    for idx, entry in enumerate(data_entries):
        # dict_keys(['note', 'hadm_id', 'problem', 'label', 'panel_1', 'final'])
        logger.info("Processing item %d...", idx)
        processed_indices.append(idx)
        tasks.append(process_with_semaphore(entry["note"], entry["problem"]))

    all_results = await asyncio.gather(*tasks)
    
    for idx, result in zip(processed_indices, all_results):
        if result:
            data_entries[idx].update(result)

    return data_entries

# 1) Our new function that processes a single JSON file
async def process_file_baseline(input_json: str, output_json: str) -> None:
    # Load input data
    logger.info(f"Loading file {input_json}")
    with open(input_json, "r") as f:
        data = json.load(f)

    # Process
    logger.info("Processing baseline queries for %d items.", len(data))
    new_data = await process_multiple_queries_baseline(data, concurrency=10)

    # Write partial results
    with open(output_json, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved baseline results to %s.", output_json)

    # Optionally re-run items with "Unknown" results
    unknown_indices = [
        i for i, entry in enumerate(new_data) 
        if entry.get("BaselineChoice") == "Unknown"
    ]
    if unknown_indices:
        logger.info("Found %d items with 'Unknown' results. Re-running those...", len(unknown_indices))

        sem = asyncio.Semaphore(10)
        async def process_unknown(idx: int) -> Optional[Dict[str, Any]]:
            async with sem:
                return await process_single_query_baseline(
                    new_data[idx]["note"], new_data[idx]["problem"]
                )

        tasks = [process_unknown(i) for i in unknown_indices]
        unknown_results = await asyncio.gather(*tasks)

        for idx, result in zip(unknown_indices, unknown_results):
            if result:
                new_data[idx].update(result)

        # Save again
        with open(output_json, "w") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        logger.info("Finished reprocessing 'Unknown' items for %s.", input_json)

# 2) main() that runs the baseline for multiple files in parallel
async def main():
    # Input -> Output files
    file_pairs = [
        (
            "/home/yl3427/cylab/SOAP_MA/Output/SOAP/3_problems_acute_kidney_injury.json",
            "/home/yl3427/cylab/SOAP_MA/Output/SOAP/3_problems_acute_kidney_injury_baseline.json"
        ),
        (
            "/home/yl3427/cylab/SOAP_MA/Output/SOAP/3_problems_sepsis.json",
            "/home/yl3427/cylab/SOAP_MA/Output/SOAP/3_problems_sepsis_baseline.json"
        ),
        (
            "/home/yl3427/cylab/SOAP_MA/Output/SOAP/3_problems_congestive_heart_failure.json",
            "/home/yl3427/cylab/SOAP_MA/Output/SOAP/3_problems_congestive_heart_failure_baseline.json"
        ),
    ]

    # Create tasks for each input-output pair
    tasks = []
    for (input_json_path, output_json_path) in file_pairs:
        tasks.append(asyncio.create_task(process_file_baseline(input_json_path, output_json_path)))

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler('log/0408_baseline_parallel_run.log', mode='a'),
            logging.StreamHandler()
        ]
    )
    asyncio.run(main())
