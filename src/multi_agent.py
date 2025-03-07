import json
import pickle
from typing import List, Dict, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field, create_model
import pandas as pd
from openai import OpenAI
import re

system_instruction = """
You are a knowledgeable and meticulous medical expert specialized in diagnosing diseases based on partial information from SOAP notes. 
You will receive either:
1. A single-disease assessment request (“specialist” scenario), or 
2. A multiple-disease assessment request (“generalist” scenario).

In the “specialist” scenario, you focus on one disease and analyze evidence within the Subjective (S) and Objective (O) sections for or against that single disease. Your final answer must be in valid JSON with:
    {
        "reasoning": "Concise explanation of your thought process",
        "diagnosis": true_or_false
    }

In the “generalist” scenario, you must assess each disease from a given list. For each disease, identify subjective and objective evidence that supports or refutes the disease. If evidence strongly supports it, conclude the diagnosis is true; if not, conclude false. If conflicting or incomplete, offer a reasoned explanation and a likely conclusion. Your final answer must be in valid JSON with each disease as a key:
    {
      "DiseaseName1": { "reasoning": "Your reasoning...", "diagnosis": true_or_false },
      "DiseaseName2": { "reasoning": "Your reasoning...", "diagnosis": true_or_false },
      ...
    }

When reasoning, consider clinical clues like symptoms, exam findings, risk factors, and labs. Clearly and succinctly justify why each disease is likely or unlikely. If any information is missing or ambiguous, note the uncertainty and choose the most probable conclusion.

Follow these instructions precisely:
• Always return output in the exact JSON format requested (no extra fields or text).
• Provide concise, medically sound rationale for each decision.
"""

prompt_specialist = """
You are a medical expert specializing in {PROBLEM}.

You are provided with only the Subjective (S) and Objective (O) sections of a patient's SOAP-formatted progress note for a potential case of {PROBLEM}.
Identify relevant clues in the subjective and objective sections that align with or argue against {PROBLEM}. If evidence strongly suggests {PROBLEM}, conclude the diagnosis is true; if not, conclude it is false. If the evidence is uncertain or conflicting, explain your reasoning and lean toward the most likely conclusion.

Patient Report:
<Subjective>
{SUBJ}
</Subjective>

<Objective>
{OBJ}
</Objective>

Your answer must be output as valid JSON formatted exactly as follows:
    {{
        "reasoning": "Your reasoning here...",
        "diagnosis": true_or_false
    }}
"""


prompt_generalist = """
You are a medical expert in diagnostic reasoning.

You are provided with only the Subjective (S) and Objective (O) sections of a patient's SOAP-formatted progress note that may be relevant to one or more of the following diseases:
{PROBLEM_LIST}

The patient may have one or more of these diseases, or none at all. Evaluate each disease independently.
Identify relevant clues in the subjective and objective sections that align with or argue against each disease. If evidence strongly suggests the disease, conclude the diagnosis is true; if not, conclude it is false. If the evidence is uncertain or conflicting, explain your reasoning and lean toward the most likely conclusion.

Patient Report:
<Subjective>
{SUBJ}
</Subjective>

<Objective>
{OBJ}
</Objective>

Your answer must be output as valid JSON formatted exactly as follows:
{{
{json_keys}
}}
"""



system_instruction_mediator = """
You are the mediator agent in a medical multi-agent diagnostic system. 
"""

class DiseaseDiagnosis(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning that leads to the final diagnosis.")
    diagnosis: bool = Field(..., description="Binary diagnosis: True if the patient has the disease, False otherwise.")



class Agent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

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

    def test_single_prob(self, dataset: pd.DataFrame, problem: str):
        pbar = tqdm(total=dataset.shape[0], desc=f"Testing {problem}")
        for idx, row in dataset.iterrows():
            subj_text = row["Subjective"]
            obj_text = row["Objective"]

            prompt_specialist_formatted = prompt_specialist.format(
                PROBLEM=problem,
                SUBJ=subj_text,
                OBJ=obj_text
            )
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_specialist_formatted}
            ]
            response = self.get_schema_followed_response(
                messages,
                schema= DiseaseDiagnosis.model_json_schema()
            )
            if response:
                dataset.at[idx, f"is_{problem.lower().replace(' ', '_')}_pred_single"] = response["diagnosis"]
                dataset.at[idx, f"is_{problem.lower().replace(' ', '_')}_reasoning_single"] = response["reasoning"]

            pbar.update(1)
        pbar.close()
        return dataset
    
    def test_multi_prob(self, dataset: pd.DataFrame, problem_lst: list):

        problem_dict = {problem: (DiseaseDiagnosis, ...) for problem in problem_lst}

        DynamicResponseMultiDiagnosis = create_model(
                    'DynamicResponseMultiDiagnosis',
                    **problem_dict
                )

        pbar = tqdm(total=dataset.shape[0], desc="Testing Multi-Diagnosis")
        for idx, row in dataset.iterrows():
            subj_text = row["Subjective"]
            obj_text = row["Objective"]

            json_keys_list = [
                f'  "{disease}": {{"reasoning": "Your reasoning here...", "diagnosis": true_or_false}}'
                for disease in problem_lst
            ]
            json_keys = ",\n".join(json_keys_list)

            prompt_generalist_formatted = prompt_generalist.format(
                PROBLEM_LIST=", ".join(problem_lst),
                SUBJ=subj_text,
                OBJ=obj_text,
                json_keys=json_keys,
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_generalist_formatted}
            ]

            response = self.get_schema_followed_response(
                messages,
                schema=DynamicResponseMultiDiagnosis.model_json_schema()
            )
            if response:
                for problem in problem_lst:
                    dataset.at[idx, f"is_{problem.lower().replace(' ', '_')}_pred_multi"] = response[problem]["diagnosis"]
                    dataset.at[idx, f"is_{problem.lower().replace(' ', '_')}_reasoning_multi"] = response[problem]["reasoning"]
            pbar.update(1)
        pbar.close()
        return dataset

if __name__ == "__main__":
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1")
    df = pd.read_csv(
        '/home/yl3427/cylab/SOAP_MA/data/mergedBioNLP2023.csv',
        usecols=['File ID', 'Subjective', 'Objective', 'Summary', 'cleaned_expanded_Summary', 'terms']
    )
    df = df.fillna('').apply(lambda x: x.str.lower())
    df['combined_summary'] = df['Summary'] + df['cleaned_expanded_Summary'] + df['terms']
    
    mi = ["myocardial infarction", "elevation mi", "non-stemi", " NSTEMI", " stemi"]
    chf = ["congestive heart failure", " chf", "HFrEF", "HFpEF"]
    pulmonary_embolism = ["pulmonary embolism"]
    pulmonary_hypertension = ["pulmonary hypertension", "pulmonary htn"]
    sepsis = ["sepsis", "septic shock"]
    urosepsis = ["urosepsis"]
    meningitis = ["meningitis"]
    aki = ["acute kidney injury", " aki", "acute renal failure", " arf"] # -> Acute tubular necrosis (ATN)인가 아닌가
    atn = ["acute tubular necrosis", " atn"]
    pancreatitis = ["pancreatitis"]
    gi_bleed = ["gastrointestinal bleed", "gi bleed"]
    hepatitis = ["hepatitis", " hep"]
    cholangitis = ["cholangitis"]
    asp_pneumonia = ["aspiration pneumonia"]

    prob_dict = {'myocardial infarction': mi, 
                 'congestive heart failure': chf, 
                 'pulmonary embolism': pulmonary_embolism, 
                 'pulmonary hypertension': pulmonary_hypertension, 
                 'sepsis': sepsis, 
                 'urosepsis': urosepsis, 
                 'meningitis': meningitis, 
                 'acute kidney injury': aki, 
                 'acute tubular necrosis': atn, 
                 'pancreatitis': pancreatitis, 
                 'gastrointestinal bleed': gi_bleed, 
                 'hepatitis': hepatitis, 
                 'cholangitis': cholangitis, 
                 'aspiration pneumonia': asp_pneumonia}

    ids = set()
    for name, lst in prob_dict.items():
        problem_terms = lst
        problem_terms = [term.lower() for term in problem_terms]

        # Use the first term as the primary term to check in the combined summary.
        primary_term = problem_terms[0]

        # Build a regex pattern that matches any of the problem terms.
        # pattern = '|'.join(problem_terms)
        pattern = '|'.join(re.escape(term) for term in problem_terms)

        mask = (
            df['combined_summary'].str.contains(pattern, na=False) &
            ~df['Subjective'].str.contains(pattern, na=False) &
            ~df['Objective'].str.contains(pattern, na=False)
        )

        filtered_df = df[mask]

        ids.update(filtered_df['File ID'])

    agent = Agent(client=client, model="meta-llama/Llama-3.3-70B-Instruct")

    df = df[df['File ID'].isin(ids)]
    df = df.reset_index(drop=True)

    result_df = agent.test_multi_prob(df, list(prob_dict.keys()))
    result_df.to_csv("multi_result_full.csv", index=False)

    for name, lst in prob_dict.items():
        result_df = agent.test_single_prob(result_df, name)
        result_df.to_csv(f"single_result_{name}.csv", index=False)
    result_df.to_csv("single_result_full.csv", index=False)