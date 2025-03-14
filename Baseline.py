import asyncio
import json
import logging
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field, create_model
from openai import AsyncOpenAI
import demjson3
import pandas as pd

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


async def main():

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


    # input_json_path = "/home/yl3427/cylab/SOAP_MA/Output/SOAP/sepsis_final.json"
    # with open(input_json_path, "r") as f:
    #     sample_data = json.load(f)
    # # sample_data = [
    # #     {"Question": "What is the best initial therapy for pneumonia?\nA) Antibiotics\nB) Surgery\nC) Radiation\nD) Physical therapy\nE) Do nothing", 
    # #      "ground_truth": "A"},
    # #     {"Question": "A patient with a headache might have:\nA) Migraine\nB) Stubbed toe\nC) Carpal tunnel\nD) Cirrhosis\nE) Diabetes",
    # #      "ground_truth": "A"},
    # # ]

    # # The multiple-choice labels might be the same for all or differ per question:
    # # choice_labels = ["A", "B", "C", "D", "E"]
    # choice_labels = ["Yes", "No"]


    #### TNM ####
    df = pd.read_csv('/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv')

    data_lst = []
    for i, row in df.iterrows():
        pathology_report = row["text"]
        ground_truth = row["t"]
        filename = row["patient_filename"]

        question_text = f"""
        Based on the following pathology report for a breast cancer patient, determine the pathologic T stage (T1, T2, T3, or T4) for breast cancer, according to the AJCC Cancer Staging Manual (7th edition). 
        Choose from T1, T2, T3, T4.

        {pathology_report}
        """
        data = {"Question": question_text, "Answer": ground_truth, "Filename": filename}
        data_lst.append(data)

    #############
    sample_data = data_lst
    choice_labels = ["T1", "T2", "T3", "T4"]
    #############



    new_data = await process_multiple_queries_baseline(sample_data, choice_labels)
    
    unknown_indices = [i for i, entry in enumerate(new_data) if entry.get("BaselineChoice") == "Unknown"]
    if unknown_indices:
        logger.info(f"Found {len(unknown_indices)} items with 'Unknown' result. Re-running those...")
        unknown_tasks = []
        for idx in unknown_indices:
            question_text = new_data[idx]["Question"]
            unknown_tasks.append(process_with_semaphore(question_text, choice_labels))
        unknown_results = await asyncio.gather(*unknown_tasks)

        # 4) Overwrite the "Unknown" results with the new results
        for idx, result in zip(unknown_indices, unknown_results):
            new_data[idx].update(result)



    output_json_path = "/home/yl3427/cylab/SOAP_MA/Output/TNM/brca_only_with_baseline.json"
    with open(output_json_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    logger.info("Finished processing all items.")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
            logging.FileHandler('log/0313_Baseline_TNM_brca_async.log', mode='w'),  # Write to file
            logging.StreamHandler()                     # Print to console
        ]
    )

    asyncio.run(main())
