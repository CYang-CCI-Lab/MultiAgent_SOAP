import sys
import os

# Determine the absolute path to the src directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(parent_dir, "src")

# Add src directory to sys.path
sys.path.insert(0, src_dir)


from prompt import *
from agent_soap import *
import pandas as pd
import pickle


if __name__ == "__main__":

    df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/data/mergedBioNLP2023.csv")

    with open("/home/yl3427/cylab/SOAP_MA/data/disease_file_dict.pkl", "rb") as file:
        disease_file_dict = pickle.load(file)

    arf_ltm_file_ids = disease_file_dict["acute renal failure"][:16]

    test_df = df[~df["File ID"].isin(arf_ltm_file_ids)]

    agent = Agent()

    # with open('/home/yl3427/cylab/SOAP_MA/ltm_per_disease/acute_renal_failure_refined.pkl', 'rb') as f:
    #    arf_ltm_refined = pickle.load(f)

    # result_df = agent.get_summary_ltm(test_df, ltm_test, arf_ltm_refined)

    # result_df.to_csv('/home/yl3427/cylab/SOAP_MA/soap_result/1114_arf_ltm.csv', index=False)

    result_df = agent.get_summary(test_df, without_ltm_test)

    result_df.to_csv(
        "/home/yl3427/cylab/SOAP_MA/soap_result/1114_arf_without_ltm.csv", index=False
    )
