# from agent import *
from agent_soap import *
import pandas as pd
import pickle

if __name__ == "__main__": 
   ltm_agent = LTMAgent()
   
   train_df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/soap_result/1031_filteredOut_ref_df.csv')[:50]
   print(len(train_df))

   ltm1 = ltm_agent.get_ltm(train_df)

   with open('/home/yl3427/cylab/SOAP_MA/soap_result/1031_raw_ltm.pkl', 'wb') as file:
      pickle.dump(ltm1, file)

   ltm2 = ltm_agent.refine_ltm()
   with open('/home/yl3427/cylab/SOAP_MA/soap_result/1031_refined_ltm.pkl', 'wb') as file:
      pickle.dump(ltm2, file)
