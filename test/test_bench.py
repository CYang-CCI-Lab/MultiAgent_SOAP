from collections import defaultdict
import pandas as pd
import json
import ast
import os
import json
import pandas as pd
import random
from typing import List, Dict, Optional, Literal
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI
import copy
from enum import Enum

with open("/home/yl3427/cylab/SOAP_MA/bench/ddxPlusDataset/release_conditions.json") as fp:
    disease_data = json.load(fp)
with open("/home/yl3427/cylab/SOAP_MA/bench/ddxPlusDataset/release_evidences.json") as fp:
    symptom_data = json.load(fp)

# ----------------------------------------------------------------------
# System instruction & Prompts
# ----------------------------------------------------------------------

system_instruction = """
You are an expert medical diagnostician with a long-term memory of general clinical reasoning principles. Please follow these rules:

1. Consider the doctor–patient dialogue carefully to identify key symptoms, history, and other relevant information.
2. Apply your general diagnostic knowledge and reasoning principles (your long-term memory) to interpret the case. Use these principles implicitly to guide your reasoning, as a real doctor would. Only mention a specific principle in your rationale if it was particularly important in making your decision.
3. Review the list of candidate diseases provided.
4. Select the single most likely disease from the provided candidates based on the case details and your clinical reasoning.
5. Provide a brief rationale for your choice, explaining why this diagnosis fits best (include any relevant general principle that influenced your decision).

IMPORTANT: 
- Your response must be valid JSON.
- Include only the following two fields in your JSON:
    1. **rationale**: a brief explanation of why you made this choice (mention any pertinent reasoning principle if applicable).
    2. **diagnosis**: the single most likely disease from the provided candidates.
"""

user_instruction = f"""
{{patient_demo}}

<Doctor-Patient Conversation>
{{dialogue}}


<Candidate Diseases>
{str(list(disease_data.keys()))}

Please provide the most likely diagnosis and your rationale in JSON, with the fields "rationale" and "diagnosis" as described.
"""


# ----------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------
DiseaseEnum = Enum(
    "DiseaseEnum", 
    ((name, name) for name in disease_data.keys()),  # (member_name, member_value)
    type=str  # make the Enum values str type for easy comparison
)
class Response(BaseModel):
    rationale: str = Field(
        description="a brief rationale for your choice"
    )
    diagnosis: DiseaseEnum = Field(
        description="the single most likely disease from the provided candidates"
    )

# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------
class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str,
    ):
        self.client = client
        self.model = model

    def get_schema_followed_response(
        self,
        messages: List[Dict],
    ) -> Optional[dict]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"guided_json": Response.model_json_schema()},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
            return None

    def parse_code(self, symptom_id):
        if '_@_' in symptom_id:
            lst = symptom_id.split('_@_')
            question = symptom_data[lst[0]]['question_en']
            if '_' in lst[1]:
                answer = symptom_data[lst[0]]['value_meaning'][lst[1]]['en']
                if answer == "Y":
                    answer = "Yes"
                elif answer == "N":
                    answer = "No"
            else:
                answer = f"{lst[1]} (out of {symptom_data[lst[0]]['possible-values'][-1]})"
        else:
            question = symptom_data[symptom_id]['question_en']
            answer = "Yes"
        return (question, answer)

    def build_dialog(self, evidence_list):
        qa_dict = defaultdict(list)
        for code in evidence_list:
            question, answer = self.parse_code(code)
            qa_dict[question].append(answer)
        dialog = [(q, a_list) for q, a_list in qa_dict.items()]
        return dialog

    def run_experiment(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:

        pbar = tqdm(total=dataset.shape[0])
        for idx, row in dataset.iterrows():
            age = row["AGE"]
            sex = "female" if row["SEX"] == "F" else "male"
            patient_demo = f"The patient is a {age}-year-old {sex}."

            evidence_list = [row['INITIAL_EVIDENCE']] + ast.literal_eval(row['EVIDENCES'])
            dialog = self.build_dialog(evidence_list)
            dialog_text = "\n".join([f"Doctor: {q}\nPatient: {', '.join(a)}." for q, a in dialog])

            formatted_user_instruction = user_instruction.format(patient_demo=patient_demo, dialogue=dialog_text)
            messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": formatted_user_instruction}
            ]
            resp = self.get_schema_followed_response(messages)

            if resp:
                dataset.at[idx, "diagnosis"] = resp['diagnosis']
                dataset.at[idx, "rationale"] = resp['rationale']
            else:
                print(f"Error in processing row {idx}")

            pbar.update(1)
        pbar.close()

        return dataset


if __name__ == "__main__":
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1", timeout=120.0)

    patient_df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/bench/ddxPlusDataset/release_validate_patients")[1000:2000]

    agent = Agent(
        client=client,
        model="meta-llama/Llama-3.3-70B-Instruct",
    )

    result_df = agent.run_experiment(patient_df)
    result_df.to_csv("/home/yl3427/cylab/SOAP_MA/bench/ddxPlusDataset/release_validate_patients_2000_0206.csv", index=False)

