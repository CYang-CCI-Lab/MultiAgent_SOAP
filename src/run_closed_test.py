from agent_soap import *
from prompt import soap_closed_so_flex, soap_closed_a_flex
import pandas as pd
from quickumls import QuickUMLS
import json

# Configuration variables
top_k = 100
reference_df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/mergedBioNLP2023.csv")
terms_df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/terms_df.csv")
lemma_terms_df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/lemma_terms_df.csv")
expanded_terms_df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/expanded_terms_df.csv")

if __name__ == "__main__": 
    agent = Agent()

    # 1. terms from raw summary
    terms_lst = terms_df['Term'].tolist()[:top_k]

    so_result_df = agent.get_closed_summary(reference_df, soap_closed_so_flex, "so", terms_lst)
    so_result_df.to_csv("/home/yl3427/cylab/SOAP_MA/soap_result/1107_soap_closed_raw_so.csv", index=False)

    a_result_df = agent.get_closed_summary(reference_df, soap_closed_a_flex, "a", terms_lst)
    a_result_df.to_csv("/home/yl3427/cylab/SOAP_MA/soap_result/1107_soap_closed_raw_a.csv", index=False)

    # 2. terms from lemmatized summary
    lemma_terms_lst = lemma_terms_df['Term'].tolist()[:top_k]

    so_result_df = agent.get_closed_summary(reference_df, soap_closed_so_flex, "so", lemma_terms_lst)
    so_result_df.to_csv("/home/yl3427/cylab/SOAP_MA/soap_result/1107_soap_closed_lemma_so.csv", index=False)

    a_result_df = agent.get_closed_summary(reference_df, soap_closed_a_flex, "a", lemma_terms_lst)
    a_result_df.to_csv("/home/yl3427/cylab/SOAP_MA/soap_result/1107_soap_closed_lemma_a.csv", index=False)

    # 3. terms from expanded summary
    expanded_terms_lst = expanded_terms_df['Term'].tolist()[:top_k]

    so_result_df = agent.get_closed_summary(reference_df, soap_closed_so_flex, "so", expanded_terms_lst)
    so_result_df.to_csv("/home/yl3427/cylab/SOAP_MA/soap_result/1107_soap_closed_expanded_so.csv", index=False)
   
    a_result_df = agent.get_closed_summary(reference_df, soap_closed_a_flex, "a", expanded_terms_lst)
    a_result_df.to_csv("/home/yl3427/cylab/SOAP_MA/soap_result/1107_soap_closed_expanded_a.csv", index=False)