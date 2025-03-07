import json
import pickle
from typing import List, Dict, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field, create_model
import pandas as pd
from openai import OpenAI

system_instruction = """
You are a medical expert in diagnostic reasoning.
"""
prompt_test_baseline = """
You are provided with only the Subjective (S) and Objective (O) sections of a new patient's SOAP-formatted progress note for a potential case of {PROBLEM}.

Patient Report:
<Subjective>
{SUBJ}
</Subjective>

<Objective>
{OBJ}
</Objective>

Based on the above patient report, provide a detailed, step-by-step reasoning of how you arrive at a conclusion. After your reasoning, provide a final binary diagnosis: "True" if the patient is diagnosed with {PROBLEM} or "False" if not.

Your answer must be output as valid JSON formatted exactly as follows:
    {{
        "reasoning": "...",
        "diagnosis": true_or_false
    }}
"""


prompt_test_diagnosis = """
You have access to long-term memory for diagnosing {PROBLEM}.
The long-term memory is as follows:
{LTM}


You are provided with only the Subjective (S) and Objective (O) sections of a new patient's SOAP-formatted progress note for a potential case of {PROBLEM}.

Patient Report:
<Subjective>
{SUBJ}
</Subjective>

<Objective>
{OBJ}
</Objective>

Based on the long-term memory and the above patient report, provide a detailed, step-by-step reasoning of how you arrive at a conclusion. After your reasoning, provide a final binary diagnosis: "True" if the patient is diagnosed with {PROBLEM} or "False" if not.

Your answer must be output as valid JSON formatted exactly as follows:
    {{
        "reasoning": "...",
        "diagnosis": true_or_false
    }}
"""


prompt_test_multi_baseline = """
You are provided with only the Subjective (S) and Objective (O) sections of a new patient's SOAP-formatted progress note that may be relevant to one or more of the following diseases:

1. {PROBLEM_1}
2. {PROBLEM_2}

Patient Report:
<Subjective>
{SUBJ}
</Subjective>

<Objective>
{OBJ}
</Objective>

Note: The patient may have one of these diseases, both, or neither. Evaluate each disease independently.

For each of the diseases, provide a detailed, step-by-step reasoning on how you arrive at a conclusion based on the patient report, and then state a final binary diagnosis ("True" if the patient is diagnosed with the disease, "False" if not).

Format your answer as valid JSON exactly as follows:
{{
  "{PROBLEM_1}": {{"reasoning": "Your reasoning here...", "diagnosis": true_or_false}},
  "{PROBLEM_2}": {{"reasoning": "Your reasoning here...", "diagnosis": true_or_false}}
}}
"""



prompt_test_multi_diagnosis = """
You have access to long-term memory for diagnosing the following diseases:

1. {PROBLEM_1}: 
{LTM_PROBLEM_1}

2. {PROBLEM_2}: 
{LTM_PROBLEM_2}


You are provided with only the Subjective (S) and Objective (O) sections of a new patient's SOAP-formatted progress note that may be relevant to one or more of the above diseases.

Patient Report:
<Subjective>
{SUBJ}
</Subjective>

<Objective>
{OBJ}
</Objective>

Note: The patient may have one of these diseases, both, or neither. Evaluate each disease independently.

For each of the diseases, provide a detailed, step-by-step reasoning based on the respective long-term memory, and then state a final binary diagnosis ("True" if the patient has the disease, "False" if not).

Your answer must be output as valid JSON formatted exactly as follows:
{{
  "{PROBLEM_1}": {{"reasoning": "Your reasoning here...", "diagnosis": true_or_false}},
  "{PROBLEM_2}": {{"reasoning": "Your reasoning here...", "diagnosis": true_or_false}}
}}
"""

class DiseaseDiagnosis(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning that leads to the final diagnosis.")
    diagnosis: bool = Field(..., description="Binary diagnosis: True if the patient has the disease, False otherwise.")



class Agent:
    def __init__(self, client: OpenAI, model: str, ltm_dict: dict):
        self.client = client
        self.model = model
        self.ltm_dict = ltm_dict

    def get_schema_followed_response(self, messages: List[Dict], schema: dict) -> Optional[dict]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"guided_json": schema},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def test_diagnosis(self, dataset: pd.DataFrame, problem: str):
        pbar = tqdm(total=dataset.shape[0], desc=f"Testing {problem}")
        for idx, row in dataset.iterrows():
            subj_text = row["Subjective"]
            obj_text = row["Objective"]

            # 1. Baseline Test
            formatted_prompt = prompt_test_baseline.format(
                PROBLEM=problem,
                SUBJ=subj_text,
                OBJ=obj_text
            )
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": formatted_prompt}
            ]
            response = self.get_schema_followed_response(
                messages,
                schema= DiseaseDiagnosis.model_json_schema()
            )
            if response:
                dataset.at[idx, f"is_{problem.lower().replace(' ', '_')}_baseline"] = response["diagnosis"]
                dataset.at[idx, f"is_{problem.lower().replace(' ', '_')}_baseline_reasoning"] = response["reasoning"]

            # 2. Single-LTM Test
            formatted_prompt = prompt_test_diagnosis.format(
                PROBLEM=problem,
                LTM=self.ltm_dict[problem],
                SUBJ=subj_text,
                OBJ=obj_text
            )
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": formatted_prompt}
            ]
            response = self.get_schema_followed_response(
                messages,
                schema= DiseaseDiagnosis.model_json_schema()
            )
            if response:
                dataset.at[idx, f"is_{problem.lower().replace(' ', '_')}_ltm"] = response["diagnosis"]
                dataset.at[idx, f"is_{problem.lower().replace(' ', '_')}_ltm_reasoning"] = response["reasoning"]
            pbar.update(1)
        pbar.close()
        return dataset
    
    def test_multi_diagnosis(self, dataset: pd.DataFrame, problem_1: str, problem_2: str):

        DynamicResponseMultiDiagnosis = create_model(
            'DynamicResponseMultiDiagnosis',
            **{
                problem_1: (DiseaseDiagnosis, ...),
                problem_2: (DiseaseDiagnosis, ...),
            }
        )

        pbar = tqdm(total=dataset.shape[0], desc="Testing Multi-Diagnosis")
        for idx, row in dataset.iterrows():
            subj_text = row["Subjective"]
            obj_text = row["Objective"]
            # 3. Multi-Baseline Test
            formatted_prompt = prompt_test_multi_baseline.format(
                PROBLEM_1=problem_1,
                PROBLEM_2=problem_2,
                SUBJ=subj_text,
                OBJ=obj_text
            )
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": formatted_prompt}
            ]
            response = self.get_schema_followed_response(
                messages,
                schema=DynamicResponseMultiDiagnosis.model_json_schema()
            )
            if response:
                dataset.at[idx, f"is_{problem_1.lower().replace(' ', '_')}_baseline"] = response[problem_1]["diagnosis"]
                dataset.at[idx, f"is_{problem_1.lower().replace(' ', '_')}_baseline_reasoning"] = response[problem_1]["reasoning"]
                dataset.at[idx, f"is_{problem_2.lower().replace(' ', '_')}_baseline"] = response[problem_2]["diagnosis"]
                dataset.at[idx, f"is_{problem_2.lower().replace(' ', '_')}_baseline_reasoning"] = response[problem_2]["reasoning"]
            # 4. Multi-LTM Test
            formatted_prompt = prompt_test_multi_diagnosis.format(
                PROBLEM_1=problem_1,
                PROBLEM_2=problem_2,
                LTM_PROBLEM_1=self.ltm_dict[problem_1],
                LTM_PROBLEM_2=self.ltm_dict[problem_2],
                SUBJ=subj_text,
                OBJ=obj_text
            )
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": formatted_prompt}
            ]
            response = self.get_schema_followed_response(
                messages,
                schema=DynamicResponseMultiDiagnosis.model_json_schema()
            )
            if response:
                dataset.at[idx, f"is_{problem_1.lower().replace(' ', '_')}_ltm"] = response[problem_1]["diagnosis"]
                dataset.at[idx, f"is_{problem_1.lower().replace(' ', '_')}_ltm_reasoning"] = response[problem_1]["reasoning"]
                dataset.at[idx, f"is_{problem_2.lower().replace(' ', '_')}_ltm"] = response[problem_2]["diagnosis"]
                dataset.at[idx, f"is_{problem_2.lower().replace(' ', '_')}_ltm_reasoning"] = response[problem_2]["reasoning"]
            pbar.update(1)
        pbar.close()
        return dataset

if __name__ == "__main__":
    # Initialize API client and agent
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1")
    
    # Load the dataset and split information
    df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/data/mergedBioNLP2023_interest.csv')

    terms_interest = ['hypertension','acute renal failure',
                      'atrial fibrillation','anemia','coronary artery disease',
                      'hypotension', 'altered mental status', 'respiratory failure',
                      'diabetes mellitus', 'leukocytosis']
    
    ltm_dict = {}
    for term in terms_interest:
        with open(f"/home/yl3427/cylab/SOAP_MA/ltm_per_disease/0212_memory_{term}.pkl", "rb") as file:
            ltm = pickle.load(file)
            ltm_dict[term] = ltm
    
    agent = Agent(client=client, model="meta-llama/Llama-3.3-70B-Instruct", ltm_dict=ltm_dict)

    with open('/home/yl3427/cylab/SOAP_MA/data/split.pkl', 'rb') as f:
        split = pickle.load(f)

    for term in terms_interest:
        train_ids = split[term][:10]
        # rows that are not in train_ids
        test_df = df[~df['File ID'].isin(train_ids)]
        results = agent.test_diagnosis(test_df, term)
        results.to_csv(f'/home/yl3427/cylab/SOAP_MA/data/0212_results_single_{term}.csv', index=False)
    
    for term_idx in range(0, len(terms_interest)-1, 2):
        problem_1 = terms_interest[term_idx]
        problem_2 = terms_interest[term_idx + 1]
        train_ids_total = split[problem_1][:10] + split[problem_2][:10]
        test_df = df[~df['File ID'].isin(train_ids_total)]
        results = agent.test_multi_diagnosis(test_df, problem_1, problem_2)
        results.to_csv(f'/home/yl3427/cylab/SOAP_MA/data/0212_results_multi_{problem_1}_{problem_2}.csv', index=False)
