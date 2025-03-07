import json
import pickle
from typing import List, Dict, Optional, Literal
from tqdm import tqdm
from pydantic import BaseModel, Field, create_model
import pandas as pd
from openai import OpenAI
import re
from typing import Optional, Union, List, get_origin, get_args, Any
import inspect

# Initialize OpenAI Client
client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1")

# Generic LLM agent class
class LLMAgent:
    def __init__(self, client: OpenAI, system_prompt: str):
        self.client = client
        self.model = self.client.models.list().data[0].id
        self.messages = [{"role": "system", "content": system_prompt}]

    def get_response(self, user_prompt: str, guided_: dict = None, tools: List[dict] = None):
        self.messages.append({"role": "user", "content": user_prompt})

        params = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0.3,
        }
        if guided_:
            params["extra_body"] = guided_ # {"guided_json": json_schema}, {"guided_choice": ["A", "B", "C", "D", "E"]}
        if tools:
            params["tools"] = tools

        response = self.client.chat.completions.create(**params)
        if response.choices[0].message.tool_calls:
            # Not implemented yet
            pass
        if response.choices[0].message.content:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message

# Generalist Agent
class GeneralistAgent(LLMAgent):
    def identify_specialists(self, task_context: str, choices: List[str]):
        class Specialist(BaseModel):
            specialist: str = Field(..., description="Role of the specialist")
            expertise: List[str] = Field(..., description="Expertise of the specialist")
        
        panel_dict = {f"Specialist_{i+1}": (Specialist, ...) for i in range(5)}
        SpecialistPanel = create_model("SpecialistPanel", **panel_dict)
      
        prompt = f"""
        Given the following query:
        {task_context}

        Possible answer options:
        {', '.join(choices)}

        Identify 5 relevant medical specialties needed to accurately address the query.
        Return a JSON mapping each specialist's field to their expertise.
        """
        message = self.get_response(prompt, guided_={"guided_json": SpecialistPanel.model_json_schema()})
        try:
            specialists = json.loads(message.content)
            print("Identified specialists:", specialists)
            return specialists
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Raw response:", message)
            return None

    def make_final_decision(self, task_context: str, specialist_responses: Dict[str, Any], choices: List[str]) -> Any:
        prompt = f"""
        Based on the original query:
        {task_context}

        We've hired the specialists you recommended. Here are their detailed analyses:
        {json.dumps(specialist_responses, indent=2)}

        Review the analyses and choose the final answer from these options:
        {', '.join(choices)}

        Provide step-by-step reasoning before giving the final decision.
        """
        choices_tup = tuple(choices)

        class DiseaseDiagnosis(BaseModel):
            reasoning: str = Field(..., description="Step-by-step reasoning that leads to the final choice")
            choice: Literal[choices_tup] = Field(..., description="Final choice")

        return self.get_response(prompt, guided_={"guided_json": DiseaseDiagnosis.model_json_schema()})

# Specialist Generator

def specialist_generator(field: str, expertise: str) -> LLMAgent:
    system_prompt = (
        f"You are a medical expert in {field}.\n"
        f"Your expertise includes:\n{expertise}\n"
    )
    return LLMAgent(client, system_prompt=system_prompt)

# Main workflow

def multi_agent_medical_analysis(task_context: str, choices: List[str], task: Literal["QA", "Diagnosis"]):

    QA_SYSTEM_PROMPT = "You are a clinical assistant tasked with answering multiple-choice questions about medical knowledge."
    DIAGNOSIS_SYSTEM_PROMPT = "You are a generalist medical practitioner who coordinates diagnoses and consultations."

    if task == "QA":
        system_instruction = QA_SYSTEM_PROMPT
    else:
        system_instruction = DIAGNOSIS_SYSTEM_PROMPT

    # Step 1: Initialize Generalist Agent
    generalist = GeneralistAgent(client, system_prompt=system_instruction)

    # Generalist identifies specialists
    specialists_info = generalist.identify_specialists(task_context, choices)

    # Step 2: Create and query Specialist Agents
    specialist_responses = {}
    for _, specialist in specialists_info.items():
        print(f"Crating specialist agent for {specialist['specialist']}...")
        specialist_agent = specialist_generator(specialist["specialist"], '\n'.join(specialist["expertise"]))
        response = specialist_agent.get_response(f"Analyze this from the perspective of a {specialist['specialist']}: \n{task_context}")
        specialist_responses[specialist["specialist"]] = response.content
        print(f"Specialist ({specialist['specialist']}) response:", response.content)

    # Step 3: Generalist makes final decision
    final_decision_mssg = generalist.make_final_decision(task_context, specialist_responses, choices)
    try:
        final_decision = json.loads(final_decision_mssg.content)
        return final_decision
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Raw response:", final_decision_mssg)
        return None

# Example usage
if __name__ == "__main__":
    TASK = ["QA", "Diagnosis"][1]
    if TASK == "QA":
        qa_df = pd.read_csv("/home/yl3427/cylab/llm_reasoning/reasoning/data/step1a.csv", encoding="latin-1") # idx,question,choice,ground_truth,qn_num
        for i, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Processing QA Data"):
            question = row['question']
            choice = row['choice']
            qa_context = f"""
            Question: {question}
            Choices: {choice}
            """
            result = multi_agent_medical_analysis(task_context=qa_context, choices=["A", "B", "C", "D", "E"], task = TASK)
            if not result:
                continue
            qa_df.at[i, "pred"] = result['choice']
            qa_df.at[i, "reasoning"] = result['reasoning']
            print(f"Question {i+1} - Prediction: {result['choice']}")
            print(f"Reasoning: {result['reasoning']}")
        qa_df.to_csv("/home/yl3427/cylab/SOAP_MA/MA_results/step1a_pred2.csv", index=False)

    elif TASK == "Diagnosis":
        df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/data/mergedBioNLP2023.csv',
                              usecols=['File ID', 'Subjective', 'Objective', 'Assessment', 'Summary', 'cleaned_expanded_Summary', 'terms'])

        df = df.fillna('').apply(lambda x: x.str.lower())
        df['combined_summary'] = df['Summary'] + df['cleaned_expanded_Summary'] + df['terms']

        mi = ["myocardial infarction", "elevation mi", "non-stemi", " NSTEMI", " stemi"]
        chf = ["congestive heart failure", " chf", "HFrEF", "HFpEF"]
        pulmonary_embolism = ["pulmonary embolism"]
        pulmonary_hypertension = ["pulmonary hypertension", "pulmonary htn"]
        sepsis = ["sepsis", "septic shock"]
        urosepsis = ["urosepsis"]
        meningitis = ["meningitis"]
        aki = ["acute kidney injury", " aki", "acute renal failure", " arf"] # -> Acute tubular necrosis (ATN)인가 아닌가
        atn = ["acute tubular necrosis", " atn"]
        pancreatitis = ["pancreatitis"]
        gi_bleed = ["gastrointestinal bleed", "gi bleed"]
        hepatitis = ["hepatitis", " hep"]
        cholangitis = ["cholangitis"]
        asp_pneumonia = ["aspiration pneumonia"]

        prob_dict = {'sepsis': sepsis, 
                        'acute kidney injury': aki, 
                        'pancreatitis': pancreatitis, 
                        'gastrointestinal bleed': gi_bleed,
                        "congestive heart failure": chf}

        ids = set()
        for name, lst in prob_dict.items():
            problem_terms = lst
            problem_terms = [term.lower() for term in problem_terms]

            # Use the first term as the primary term to check in the combined summary.
            primary_term = problem_terms[0]

            # Build a regex pattern that matches any of the problem terms.
            # pattern = '|'.join(problem_terms)
            pattern = '|'.join(re.escape(term) for term in problem_terms)

            mask = (
                df['combined_summary'].str.contains(pattern, na=False) &
                ~df['Subjective'].str.contains(pattern, na=False) &
                ~df['Objective'].str.contains(pattern, na=False)
            )

            filtered_df = df[mask]

            ids.update(filtered_df['File ID'])

        df = df[df['File ID'].isin(ids)]
        # diag_df = df.reset_index(drop=True)[137:]
        diag_df = pd.read_csv('/home/yl3427/cylab/SOAP_MA/MA_results/diag_0302.csv')


        print("Let's start the diagnosis process...")
        for problem in prob_dict.keys():
            for i, row in tqdm(diag_df.iterrows(), total=len(diag_df), desc="Processing Diagnosis Data"):
                if not pd.isna(row['pred_sepsis']):
                    continue
                subj = row['Subjective']
                obj = row['Objective']
                diag_context = f"""
                Based on the below patient report, does the patient have {problem}?
                {subj}
                {obj}
                """
                result = multi_agent_medical_analysis(task_context=diag_context, choices=["Yes", "No"], task = TASK)
                if not result:
                    continue
                diag_df.at[i, f"pred_{problem}"] = result['choice']
                diag_df.at[i, f"reasoning_{problem}"] = result['reasoning']
                print(f"Report {i+1} - Does the patient have {problem}? {result['choice']}")
                print(f"Reasoning: {result['reasoning']}")
                diag_df.to_csv(f"/home/yl3427/cylab/SOAP_MA/MA_results/diag_0307_missing.csv", index=False)
