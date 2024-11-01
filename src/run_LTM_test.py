from prompt import *
from agent_soap import *
import pandas as pd
import pickle
from quickumls import QuickUMLS

if __name__ == "__main__": 
   agent = Agent()
   
   test_df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/soap_result/1031_filtered_ref_df.csv')

   with open('/home/yl3427/cylab/SOAP_MA/soap_result/1031_refined_ltm.pkl', 'rb') as file:
      ltm = pickle.load(file)

   # ltm_open_df = agent.get_summary_ltm(test_df, ltm_test_open, "so", ltm)
   # ltm_open_df.to_csv('/home/yl3427/cylab/SOAP_MA/soap_result/1031_ltm_open.csv', index=False)

  


   # a pool of terms
   train_df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/BioNLP2023-1A-Train.csv')
   test_df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/BioNLP2023-1A-Test.csv')
   reference_df = pd.concat([train_df, test_df], ignore_index=True)
   filtered_matcher = QuickUMLS('/home/yl3427/cylab/QuickUMLS', 
                    overlapping_criteria="length",
                    threshold=0.8,
                    accepted_semtypes=['T037', 'T046', 'T047', 'T048', 'T049', 'T190', 'T191'],
                    # accepted_semtypes=['T047']
                    )
   
   valid_fileids = set()
   pool_of_terms = set()
   reference_df['Summary2'] = reference_df['Summary'].apply(lambda x: "")
   for i, row in reference_df.iterrows():
      summary = row['Summary']
      new_summary = summary

      try:
         sum_lst= summary.split(";")
      except:
         continue

      matched_sum_lst = [filtered_matcher.match(s, best_match=True) for s in sum_lst]
      if not all(matched_sum_lst):
         continue

      for s in sum_lst:
         matches = filtered_matcher.match(s, best_match=True)
         term = sorted(matches[0], key=lambda x: x['similarity'], reverse=True)[0]['term']
         new_summary = new_summary.replace(s, term)
         pool_of_terms.add(term)

      reference_df.at[i, 'Summary2'] = new_summary
      valid_fileids.add(row['File ID'])

   terms_lst = list(pool_of_terms)

   test_df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/soap_result/1031_filtered_ref_df.csv')
   ltm_close_df = agent.get_closed_summary_ltm(test_df, ltm_test_closed, "so", terms_lst, ltm)
   ltm_close_df.to_csv('/home/yl3427/cylab/SOAP_MA/soap_result/1031_ltm_close.csv', index=False)