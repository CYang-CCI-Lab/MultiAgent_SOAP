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


df = pd.read_csv(
    "/home/yl3427/cylab/SOAP_MA/soap_result/1206_hypertension_so_a_mixtral.csv"
)

client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1", timeout=120.0)


if __name__ == "__main__":
    test_agent = TestAgent(
        client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1", disease="_"
    )
    results = test_agent.test_with_assessment(df)
    results.to_csv(
        f"/home/yl3427/cylab/SOAP_MA/soap_result/1206_hypertension_so_a_mixtral.csv",
        index=False,
    )
