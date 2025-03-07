import json
import pickle
from typing import List, Dict, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field
import pandas as pd
from openai import OpenAI
import ast
import numpy as np
from collections import defaultdict

system_instruction = """
You are a medical expert in diagnostic reasoning. 
"""

prompt_memory_generate = """
You are provided with sections of a SOAP-formatted progress note for a patient diagnosed with {PROBLEM}, specifically the Subjective (S), Objective (O) and Assessment (A) sections.

Identify patterns in how medical experts might have diagnosed {PROBLEM} based on the information provided in these sections. Generate “memory items” that capture these patterns so they can be used to aid in diagnosing future patients with {PROBLEM}.

Progress Note of a Patient with {PROBLEM}:
<Subjective>
{subjective_section}
</Subjective>

<Objective>
{objective_section}
</Objective>

<Assessment>
{assessment_section}
</Assessment>

Structure your findings as valid JSON, formatted as follows, containing only the field:
    {{
        "memory": [
            ...memory items...
        ]
    }}
"""


prompt_memory_refinement = """
Below is the current accumulated memory derived from multiple cases of diagnosing {PROBLEM}:

{all_accumulated_memory}

Please refine this memory by removing any redundancies, contradictions, or irrelevant details.

Output the refined memory as valid JSON with a single field:
    {{
        "memory": [
            ...refined memory items...
        ]
    }}
"""




class ResponseMemory(BaseModel):
    memory: List[str] = Field(
        default_factory=list,
        description="A list of memory items that capture patterns in diagnosing a specific disease."
    )


class Agent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.memory: List[str] = []

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

    def build_memory(self, dataset: pd.DataFrame, problem):
        pbar = tqdm(total=dataset.shape[0], desc="Processing Notes")

        for idx, row in dataset.iterrows():
            subjective_text = row["Subjective"]
            objective_text = row["Objective"]
            assessment_text = row["Assessment"]

            formatted_prompt = prompt_memory_generate.format(
                PROBLEM=problem,
                subjective_section=subjective_text,
                objective_section=objective_text,
                assessment_section=assessment_text
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": formatted_prompt}
            ]

            response = self.get_schema_followed_response(
                messages, 
                schema=ResponseMemory.model_json_schema()
            )
            if response:
                self.memory.extend(response["memory"])

            pbar.update(1)
        pbar.close()
      
        self.refine_long_term_memory(problem=problem)
        self.refine_long_term_memory(problem=problem)
        self.refine_long_term_memory(problem=problem)
        self.refine_long_term_memory(problem=problem)
        self.refine_long_term_memory(problem=problem)
        with open(f"/home/yl3427/cylab/SOAP_MA/ltm_per_disease/0212_memory_{problem}.pkl", "wb") as f:
            pickle.dump(self.memory, f)

        self.memory.clear()

    def refine_long_term_memory(self, problem):
        formatted_prompt = prompt_memory_refinement.format(
            PROBLEM=problem,
            all_accumulated_memory=str(self.memory)
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": formatted_prompt}
        ]

        response = self.get_schema_followed_response(
            messages,
            schema=ResponseMemory.model_json_schema()
        )
        if response:
            self.memory = response["memory"]
        

    


if __name__ == "__main__":
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1")

    agent = Agent(
        client=client,
        model="meta-llama/Llama-3.3-70B-Instruct",
    )
    df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/data/mergedBioNLP2023_interest.csv')

    # terms_interest = ['hypertension','acute renal failure',
    #                   'atrial fibrillation','anemia','coronary artery disease',
    #                   'hypotension', 'altered mental status', 'respiratory failure',
    #                   'diabetes mellitus', 'leukocytosis']
    terms_interest = ['acute renal failure', 'altered mental status', 'respiratory failure']
   
    with open('/home/yl3427/cylab/SOAP_MA/data/split.pkl', 'rb') as f:
        split = pickle.load(f)
    
    for term in terms_interest:
        train_ids = split[term][:10]
        train_df = df[df['File ID'].isin(train_ids)]
        agent.build_memory(train_df, term)


