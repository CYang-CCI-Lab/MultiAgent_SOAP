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


class Agent:
    def __init__(self):
        self.client = OpenAI()

    def get_response(self, prompt):
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert in diagnostic reasoning.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06", messages=messages, temperature=0
        )

        return response.choices[0].message.content

    def get_summary(
        self, df, prompt, input_type: Literal["soa", "so", "sa", "a"] = "so"
    ):
        parsing_error = 0
        pbar = tqdm(total=df.shape[0])

        for idx, row in df.iterrows():
            subj = row["Subjective"]
            obj = row["Objective"]
            ass = row["Assessment"]

            if input_type.lower() == "soa":
                formatted_prompt = prompt.format(
                    subjective_section=subj,
                    objective_section=obj,
                    assessment_section=ass,
                )
            elif input_type.lower() == "so":
                formatted_prompt = prompt.format(
                    subjective_section=subj, objective_section=obj
                )
            elif input_type.lower() == "sa":
                formatted_prompt = prompt.format(
                    subjective_section=subj, assessment_section=ass
                )
            else:
                formatted_prompt = prompt.format(assessment_section=ass)

            response = self.get_response(formatted_prompt)

            try:
                df.at[idx, f"pred"] = response
            except Exception as e:
                print(f"An error occurred: {e}")
                parsing_error += 1
            pbar.update(1)
        pbar.close()
        print(f"Total parsing errors: {parsing_error}")
        return df

    def get_summary_ltm(
        self, df, prompt, ltm, input_type: Literal["soa", "so", "sa", "a"] = "so"
    ):
        parsing_error = 0
        pbar = tqdm(total=df.shape[0])

        for idx, row in df.iterrows():
            subj = row["Subjective"]
            obj = row["Objective"]
            ass = row["Assessment"]

            if input_type.lower() == "soa":
                formatted_prompt = prompt.format(
                    list_of_rules=ltm,
                    subjective_section=subj,
                    objective_section=obj,
                    assessment_section=ass,
                )
            elif input_type.lower() == "so":
                formatted_prompt = prompt.format(
                    list_of_rules=ltm, subjective_section=subj, objective_section=obj
                )
            elif input_type.lower() == "sa":
                formatted_prompt = prompt.format(
                    list_of_rules=ltm, subjective_section=subj, assessment_section=ass
                )
            else:
                formatted_prompt = prompt.format(
                    list_of_rules=ltm, assessment_section=ass
                )

            response = self.get_response(formatted_prompt)

            try:
                df.at[idx, f"pred"] = response
            except Exception as e:
                print(f"An error occurred: {e}")
                parsing_error += 1
            pbar.update(1)
        pbar.close()
        print(f"Total parsing errors: {parsing_error}")
        return df

    def get_closed_summary(
        self, df, prompt, input_type: Literal["soa", "so", "sa", "a"], terms_lst
    ):
        parsing_error = 0
        pbar = tqdm(total=df.shape[0])

        for idx, row in df.iterrows():
            subj = row["Subjective"]
            obj = row["Objective"]
            ass = row["Assessment"]

            if input_type.lower() == "soa":
                formatted_prompt = prompt.format(
                    list_of_terms=", ".join(terms_lst),
                    subjective_section=subj,
                    objective_section=obj,
                    assessment_section=ass,
                )
            elif input_type.lower() == "so":
                formatted_prompt = prompt.format(
                    list_of_terms=", ".join(terms_lst),
                    subjective_section=subj,
                    objective_section=obj,
                )
            elif input_type.lower() == "sa":
                formatted_prompt = prompt.format(
                    list_of_terms=", ".join(terms_lst),
                    subjective_section=subj,
                    assessment_section=ass,
                )
            else:
                formatted_prompt = prompt.format(
                    list_of_terms=", ".join(terms_lst), assessment_section=ass
                )

            response = self.get_response(formatted_prompt)

            try:
                df.at[idx, f"pred"] = response
            except Exception as e:
                print(f"An error occurred: {e}")
                parsing_error += 1
            pbar.update(1)
        pbar.close()
        print(f"Total parsing errors: {parsing_error}")
        return df

    def get_closed_summary_ltm(
        self, df, prompt, input_type: Literal["soa", "so", "sa", "a"], terms_lst, ltm
    ):
        parsing_error = 0
        pbar = tqdm(total=df.shape[0])

        for idx, row in df.iterrows():
            subj = row["Subjective"]
            obj = row["Objective"]
            ass = row["Assessment"]

            if input_type.lower() == "soa":
                formatted_prompt = prompt.format(
                    list_of_rules=ltm,
                    list_of_terms=", ".join(terms_lst),
                    subjective_section=subj,
                    objective_section=obj,
                    assessment_section=ass,
                )
            elif input_type.lower() == "so":
                formatted_prompt = prompt.format(
                    list_of_rules=ltm,
                    list_of_terms=", ".join(terms_lst),
                    subjective_section=subj,
                    objective_section=obj,
                )
            elif input_type.lower() == "sa":
                formatted_prompt = prompt.format(
                    list_of_rules=ltm,
                    list_of_terms=", ".join(terms_lst),
                    subjective_section=subj,
                    assessment_section=ass,
                )
            else:
                formatted_prompt = prompt.format(
                    list_of_rules=ltm,
                    list_of_terms=", ".join(terms_lst),
                    assessment_section=ass,
                )

            response = self.get_response(formatted_prompt)

            try:
                df.at[idx, f"pred"] = response
            except Exception as e:
                print(f"An error occurred: {e}")
                parsing_error += 1
            pbar.update(1)
        pbar.close()
        print(f"Total parsing errors: {parsing_error}")
        return df


class KnowledgeRules(BaseModel):
    rules: List[str]


class LTMAgent:
    def __init__(self, disease):
        self.client = OpenAI()
        self.disease = disease
        self.ltm = []

    def get_rules(self, prompt):
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert in diagnostic reasoning",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0,
            response_format=KnowledgeRules,
        )

        return response.choices[0].message.parsed.rules

    def get_ltm(self, df):
        pbar = tqdm(total=df.shape[0])
        for idx, row in df.iterrows():
            subj = row["Subjective"]
            obj = row["Objective"]
            ass = row["Assessment"]

            formatted_prompt = ltm_builder.format(
                disease=self.disease,
                subjective_section=subj,
                objective_section=obj,
                assessment_section=ass,
            )

            rules = self.get_rules(formatted_prompt)
            self.ltm.extend(rules)

            pbar.update(1)
        pbar.close()

        return self.ltm

    def refine_ltm(self):

        formatted_prompt = ltm_refiner.format(
            disease=self.disease, original_rules=self.ltm
        )
        refined_rules = self.get_rules(formatted_prompt)
        self.ltm = refined_rules

        return self.ltm
