import sys
import os

# Determine the absolute path to the src directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(parent_dir, "src")

# Add src directory to sys.path
sys.path.insert(0, src_dir)

from agent_soap import *
import pandas as pd
import pickle

if __name__ == "__main__":
    df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/data/mergedBioNLP2023.csv")

    with open("/home/yl3427/cylab/SOAP_MA/data/disease_file_dict.pkl", "rb") as file:
        disease_file_dict = pickle.load(file)

    arf_ltm_file_ids = disease_file_dict["acute renal failure"][:16]

    train_df = df[df["File ID"].isin(arf_ltm_file_ids)]

    ltm_agent = LTMAgent(disease="acute renal failure")
    ltm1 = ltm_agent.get_ltm(train_df)

    with open(
        "/home/yl3427/cylab/SOAP_MA/ltm_per_disease/acute_renal_failure_raw.pkl", "wb"
    ) as file:
        pickle.dump(ltm1, file)

    ltm2 = ltm_agent.refine_ltm()
    with open(
        "/home/yl3427/cylab/SOAP_MA/ltm_per_disease/acute_renal_failure_refined.pkl",
        "wb",
    ) as file:
        pickle.dump(ltm2, file)
