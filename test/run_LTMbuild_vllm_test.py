import sys
import os
from pydantic import BaseModel, Field

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)

from agent_soap import *
from prompt import *
import pandas as pd
import pickle


class KnowledgeRules(BaseModel):
    rules: List[str] = Field(
        description="A list of rules or pieces of knowledge that can aid in diagnosing future patients."
    )


df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/data/mergedBioNLP2023.csv")

with open("/home/yl3427/cylab/SOAP_MA/data/disease_file_dict.pkl", "rb") as file:
    disease_file_dict = pickle.load(file)

client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1", timeout=120.0)

if __name__ == "__main__":

    top_5_diseases = sorted(
        disease_file_dict.items(), key=lambda item: len(item[1]), reverse=True
    )[:5]

    for i in range(len(top_5_diseases)):
        term, file_lst = top_5_diseases[i]
        ltm_file_ids = file_lst[: int(len(file_lst) * 0.1)]
        test_file_ids_positive = file_lst[int(len(file_lst) * 0.1) :]
        test_file_ids_negative = list(
            set(df["File ID"].tolist())
            - set(ltm_file_ids)
            - set(test_file_ids_positive)
        )
        train_df = df[df["File ID"].isin(ltm_file_ids)]

        ltm_agent = LTMAgent(
            client=client,
            model="m42-health/Llama3-Med42-70B",
            schema=KnowledgeRules,
            disease=term,
        )

        prompt = prompt_template_med42.format(
            system_instruction=system_instruction, prompt=ltm_builder_vllm
        )
        # ltm_raw = ltm_agent.get_ltm(train_df, prompt)
        ltm_raw = ltm_agent.get_ltm(train_df, ltm_builder)
        print(f"Number of rules in raw LTM for {term}: {len(ltm_raw)}")
        with open(
            f"/home/yl3427/cylab/SOAP_MA/ltm_per_disease/1120_{term}_raw.pkl", "wb"
        ) as file:
            pickle.dump(ltm_raw, file)

        prompt = prompt_template_med42.format(
            system_instruction=system_instruction, prompt=ltm_refiner_vllm
        )
        ltm_refined = ltm_agent.refine_ltm(prompt)
        print(f"Number of rules in refined LTM for {term}: {len(ltm_refined)}")
        with open(
            f"/home/yl3427/cylab/SOAP_MA/ltm_per_disease/1120_{term}_refined.pkl", "wb"
        ) as file:
            pickle.dump(ltm_refined, file)
