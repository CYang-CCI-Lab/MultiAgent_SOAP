import os
from dotenv import load_dotenv
from pathlib import Path
import json
from typing import List, Dict, Union, Optional, Tuple, Literal
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from prompt import *

env_path = Path.home()
load_dotenv(dotenv_path=env_path / ".env")


class YesOrNo(BaseModel):
    answer: Literal["Yes", "No"] = Field(
        description="Concise response indicating the presence ('Yes') or absence ('No') of the condition."
    )


class ProblemList(BaseModel):
    problem_list: List[str] = Field(
        description="List of problems identified in the patient's medical record."
    )


class TestAgent:
    def __init__(self, client: OpenAI, model: str, disease: str):
        self.client = client
        self.model = model
        self.disease = disease

    def get_response(self, messages: list, schema: BaseModel):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"guided_json": schema.model_json_schema()},
                temperature=0.1,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def test_with_assessment(
        self,
        testing_df: pd.DataFrame,
        prompt: str = soap_soa,
        schema: BaseModel = ProblemList,
    ):
        parsing_error = 0
        pbar = tqdm(total=testing_df.shape[0])

        for idx, row in testing_df.iterrows():
            if testing_df.loc[idx, "is_parsed"]:
                continue
            subj = row["Subjective"]
            obj = row["Objective"]
            ass = row["generated_assess"]

            formatted_prompt = prompt.format(
                subjective_section=subj,
                objective_section=obj,
                assessment_section=ass,
            )
            messages = [{"role": "user", "content": formatted_prompt}]

            response = self.get_response(messages, schema)

            if not response:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_df.loc[idx, "is_parsed"] = False
                continue

            testing_df.loc[idx, "is_parsed"] = True
            testing_df.loc[idx, "generated_summary"] = str(response["problem_list"])

            pbar.update(1)
        pbar.close()
        print(f"Total parsing errors: {parsing_error}")
        return testing_df

    def test(
        self,
        testing_df: pd.DataFrame,
        prompt: str,
        ltm: Optional[List[str]] = None,
        schema: BaseModel = YesOrNo,
    ):
        parsing_error = 0
        pbar = tqdm(total=testing_df.shape[0])

        for idx, row in testing_df.iterrows():
            subj = row["Subjective"]
            obj = row["Objective"]

            if ltm:
                formatted_prompt = prompt.format(
                    disease=self.disease,
                    list_of_rules=ltm,
                    subjective_section=subj,
                    objective_section=obj,
                )
                column_prefix = f"{self.disease}_with_ltm"
            else:
                formatted_prompt = prompt.format(
                    disease=self.disease, subjective_section=subj, objective_section=obj
                )
                column_prefix = f"{self.disease}_without_ltm"

            messages = [{"role": "user", "content": formatted_prompt}]

            response = self.get_response(messages, schema)

            if not response:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_df.loc[idx, f"{column_prefix}_is_parsed"] = False
                continue

            testing_df.loc[idx, f"{column_prefix}_is_parsed"] = True
            testing_df.loc[idx, f"{column_prefix}_answer"] = response["answer"]

            pbar.update(1)
        pbar.close()
        print(f"Total parsing errors: {parsing_error}")
        return testing_df

    def test_cot(
        self,
        testing_df: pd.DataFrame,
        ltm: Optional[List[str]] = None,
        schema: BaseModel = YesOrNo,
    ):
        parsing_error = 0
        pbar = tqdm(total=testing_df.shape[0])

        for idx, row in testing_df.iterrows():
            subj = row["Subjective"]
            obj = row["Objective"]

            # First, generate the Assessment section
            formatted_prompt = cot1_test.format(
                subjective_section=subj, objective_section=obj
            )
            messages = [
                {
                    "role": "user",
                    "content": system_instruction + "\n" + formatted_prompt,
                }
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )
            response = response.choices[0].message.content
            messages.append({"role": "assistant", "content": response})
            testing_df.loc[idx, f"generated_assess"] = response

            if ltm:
                formatted_prompt = ltm_cot2_test.format(
                    disease=self.disease,
                    list_of_rules=ltm,
                    subjective_section=subj,
                    objective_section=obj,
                )
                column_prefix = f"{self.disease}_with_ltm"
            else:
                formatted_prompt = without_ltm_cot2_test.format(
                    disease=self.disease, subjective_section=subj, objective_section=obj
                )
                column_prefix = f"{self.disease}_without_ltm"

            messages.append({"role": "user", "content": formatted_prompt})

            response = self.get_response(messages, schema)

            if not response:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_df.loc[idx, f"{column_prefix}_is_parsed"] = False
                continue

            testing_df.loc[idx, f"{column_prefix}_is_parsed"] = True
            testing_df.loc[idx, f"{column_prefix}_answer"] = response["answer"]

            pbar.update(1)
        pbar.close()
        print(f"Total parsing errors: {parsing_error}")
        return testing_df
