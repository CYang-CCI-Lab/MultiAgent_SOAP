import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)

from prompt import *
from agent_soap import *
import pandas as pd
import pickle

from pydantic import BaseModel, Field, validator
from typing import Literal


df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/data/mergedBioNLP2023.csv")

with open("/home/yl3427/cylab/SOAP_MA/data/disease_file_dict.pkl", "rb") as file:
    disease_file_dict = pickle.load(file)

client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1", timeout=120.0)

disease_ltm_map = {}

with open(
    "/home/yl3427/cylab/SOAP_MA/ltm_per_disease/1120_hypertension_refined.pkl", "rb"
) as file:
    disease_ltm_map["hypertension"] = pickle.load(file)
with open(
    "/home/yl3427/cylab/SOAP_MA/ltm_per_disease/1120_acute renal failure_refined.pkl",
    "rb",
) as file:
    disease_ltm_map["acute renal failure"] = pickle.load(file)
with open(
    "/home/yl3427/cylab/SOAP_MA/ltm_per_disease/1120_atrial fibrillation_refined.pkl",
    "rb",
) as file:
    disease_ltm_map["atrial fibrillation"] = pickle.load(file)
with open(
    "/home/yl3427/cylab/SOAP_MA/ltm_per_disease/1120_anemia_refined.pkl", "rb"
) as file:
    disease_ltm_map["anemia"] = pickle.load(file)
with open(
    "/home/yl3427/cylab/SOAP_MA/ltm_per_disease/1120_coronary artery disease_refined.pkl",
    "rb",
) as file:
    disease_ltm_map["coronary artery disease"] = pickle.load(file)

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
        test_df = df[
            df["File ID"].isin(test_file_ids_positive + test_file_ids_negative)
        ]
        test_agent = TestAgent(
            client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1", disease=term
        )

        test_with_ltm_results = test_agent.test_cot(test_df, ltm=disease_ltm_map[term])
        test_with_ltm_results.to_csv(
            f"/home/yl3427/cylab/SOAP_MA/soap_result/1129_{term}_with_ltm_cot_mixtral.csv",
            index=False,
        )

        test_without_ltm_results = test_agent.test_cot(test_df)
        test_without_ltm_results.to_csv(
            f"/home/yl3427/cylab/SOAP_MA/soap_result/1129_{term}_without_ltm_cot_mixtral.csv",
            index=False,
        )
