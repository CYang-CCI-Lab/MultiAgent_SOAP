# from agent import *
from agent_soap import *
from prompt import (
    soap_soa_zs,
    soap_soa_os,
    soap_so_zs,
    soap_so_os,
    soap_a_zs,
    soap_a_os,
    soap_sa_zs,
    soap_sa_os,
)
import pandas as pd
import json

if __name__ == "__main__":
    agent = Agent()

    a_test_df = pd.read_csv("/home/yl3427/cylab/rag_tnm/Task-3_Test.csv")  # 87
    soap_train_df = pd.read_csv(
        "/home/yl3427/cylab/rag_tnm/BioNLP2023-1A-Train.csv"
    )  # 765, this includes all of the three above

    # # Zero-shot S, O
    # filtered_df_for_zs = as_train_df[as_train_df['File ID'].isin(set(a_test_df['File ID']).union({'190862.txt', '109943.txt', '195790.txt'}))]

    # result_df = agent.get_summary(filtered_df_for_zs, soap_so_zs, "so")
    # result_df.to_csv("/home/yl3427/cylab/rag_tnm/soap_result/1024_soap_so_zs.csv", index=False)

    # # Zero-shot A
    # filtered_df_for_zs = as_train_df[as_train_df['File ID'].isin(set(a_test_df['File ID']).union({'190862.txt', '109943.txt', '195790.txt'}))]

    # result_df = agent.get_summary(filtered_df_for_zs, soap_a_zs, "a")
    # result_df.to_csv("/home/yl3427/cylab/rag_tnm/soap_result/1024_soap_a_zs.csv", index=False)

    # # One-shot S, O
    # filtered_df_for_os = as_train_df[as_train_df['File ID'].isin(set(a_test_df['File ID']).union({'190862.txt', '109943.txt', '195790.txt'}))]

    # result_df = agent.get_summary(filtered_df_for_os, soap_so_os, "so")
    # result_df.to_csv("/home/yl3427/cylab/rag_tnm/soap_result/1024_soap_so_os.csv", index=False)

    # # One-shot A
    # filtered_df_for_os = as_train_df[as_train_df['File ID'].isin(set(a_test_df['File ID']).union({'190862.txt', '109943.txt', '195790.txt'}))]

    # result_df = agent.get_summary(filtered_df_for_os, soap_a_os, "a")
    # result_df.to_csv("/home/yl3427/cylab/rag_tnm/soap_result/1024_soap_a_os.csv", index=False)

    filtered_df_for_zs = soap_train_df[
        soap_train_df["File ID"].isin(
            set(a_test_df["File ID"]).union({"190862.txt", "109943.txt", "195790.txt"})
        )
    ]

    result_df = agent.get_summary(filtered_df_for_zs, soap_sa_zs, "sa")
    result_df.to_csv(
        "/home/yl3427/cylab/rag_tnm/soap_result/1024_soap_sa_zs.csv", index=False
    )

    ####

    filtered_df_for_os = soap_train_df[
        soap_train_df["File ID"].isin(
            set(a_test_df["File ID"]).union({"190862.txt", "109943.txt", "195790.txt"})
        )
    ]

    result_df = agent.get_summary(filtered_df_for_os, soap_sa_os, "sa")
    result_df.to_csv(
        "/home/yl3427/cylab/rag_tnm/soap_result/1024_soap_sa_os.csv", index=False
    )
