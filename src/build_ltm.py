import json
import pickle
from typing import List, Dict, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field
import pandas as pd
from openai import OpenAI

# ------------------------------------------------------
# System and Prompt Templates
# ------------------------------------------------------
system_instruction = """
You are a medical expert in diagnostic reasoning. 
You rely on SOAP (Subjective, Objective, Assessment, Plan) notes to guide your reasoning process and produce evidence-based, clinically relevant responses.
"""

prompt_generate_assess = """
Below is the patient’s Subjective (S) and Objective (O) information:

Subjective:
{subjective}

Objective:
{objective}

Based on this information, please generate a concise and medically accurate Assessment. 
Include likely diagnoses, potential etiologies, and relevant factors to consider.
"""

prompt_reflect = """
The actual ground truth {data_type} is: {ground_truth}

Reflect on the differences between your generated {data_type} and this ground truth. 
Identify specific gaps, discrepancies, or misunderstandings, and briefly explain why those might have occurred.
"""

prompt_lesson = """ 
From the gaps you identified, please generate lessons learned to improve future diagnostic reasoning.

Specifically, produce two types of lessons:
1. General clinical reasoning principles that apply to all patient conditions and clinical scenarios, regardless of any specific disease.
2. Disease- or condition-specific insights.
"""

prompt_generate_summ = """
Based on the information you have seen so far, please summarize the key Problems or Diagnoses in a concise list.
"""

prompt_memory_refinement = """
Below is the current accumulated memory (lessons) from previous notes:

{all_accumulated_memory}

Please refine this memory by:
1. Removing any redundancies or contradictions.
2. Keeping only the most insightful additions that address common pitfalls or borderline cases.
3. Returning a cleaned-up, concise set of bullet points or short statements.
"""

# ------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------
class ResponseAssessment(BaseModel):
    assessment: str = Field(
        description=(
            "The text of the Assessment based on the provided Subjective and Objective information."
        )
    )

class ResponseSummary(BaseModel):
    problems: str = Field(
        description=(
            "A concise list or description of the problems or diagnoses derived "
            "from the Subjective, Objective, and Assessment sections."
        )
    )

class ResponseGap(BaseModel):
    gap: str = Field(
        description=(
            "The lesson(s) or insights learned from the mismatch "
            "between your output and the ground truth. If there is no mismatch or if you believe "
            "the ground truth is incorrect, this can be an empty string."
        )
    )

class ResponseLesson(BaseModel):
    general: List[str] = Field(
        description=(
            "A set of general lessons (not specific to any particular disease)."
        )
    )
    disease_specific: List[str] = Field(
        description=(
            "A set of lessons focusing on a specific disease or condition."
        )
    )

class ResponseMemoryRefinement(BaseModel):
    refined_memory: List[str] = Field(
        description=(
            "A cleaner, more authoritative set of memory items after removing redundancies, "
            "contradictions, or confusion from the previously accumulated lessons."
        )
    )

# Schema set for guided JSON
schema_set = {
    "summary_generation": ResponseSummary.model_json_schema(),
    "assessment_generation": ResponseAssessment.model_json_schema(),
    "gap_generation": ResponseGap.model_json_schema(),
    "lesson_generation": ResponseLesson.model_json_schema(),
    "memory_refinement": ResponseMemoryRefinement.model_json_schema(),
}


class Agent:
    """
    An Agent that processes a DataFrame of SOAP notes to build
    and refine a "memory" of general lessons and disease-specific lessons.
    
    Attributes:
        client (OpenAI): The OpenAI client used to generate responses.
        model (str): The LLM model name used in the generation calls.
        test_name (str): Tag used to name the output files containing memory checkpoints.
        general_memory (List[str]): Accumulated general lessons learned.
        specific_memory (List[str]): Accumulated disease-specific lessons learned.
    """
    def __init__(self, client: OpenAI, model: str, test_name: str) -> None:
        self.client = client
        self.model = model
        self.test_name = test_name
        self.general_memory: List[str] = []
        self.specific_memory: List[str] = []

    def get_schema_followed_response(
        self, 
        messages: List[Dict], 
        schema: dict
    ) -> Optional[dict]:
        """
        Sends a request to the LLM with guided JSON to ensure output follows a specific schema.

        Args:
            messages (List[Dict]): The chat message history to send to the LLM.
            schema (dict): JSON schema specifying how the LLM output should be structured.

        Returns:
            Optional[dict]: A dictionary parsed from the LLM response if successful, else None.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"guided_json": schema},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def build_memory(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates over each row of the provided dataset, generates and refines 
        assessments, problem lists, and lessons, accumulating them in 
        self.general_memory and self.specific_memory.

        Args:
            dataset (pd.DataFrame): Must contain 'Subjective', 'Objective', 
                                    'Assessment', 'cleaned_expanded_Summary' columns.

        Returns:
            pd.DataFrame: Updated memory (general, specific).
        """
        parsing_error = 0
        pbar = tqdm(total=dataset.shape[0], desc="Processing Notes")

        for idx, row in dataset.iterrows():
            subjective_text = row["Subjective"]
            objective_text = row["Objective"]
            ground_truth_assessment = row["Assessment"]
            ground_truth_summary = row["cleaned_expanded_Summary"]

            # ----------------------------------------
            # 1. Generate Assessment
            # ----------------------------------------
            formatted_prompt = prompt_generate_assess.format(
                subjective=subjective_text,
                objective=objective_text
            )
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": formatted_prompt}
            ]
            ass_output = self.get_schema_followed_response(
                messages, 
                schema_set["assessment_generation"]
            )

            if not ass_output:
                parsing_error += 1
                print(f"Error in generating Assessment at index: {idx}")
                pbar.update(1)
                continue

            generated_assessment = ass_output["assessment"]
            # Append the model's generated assessment to the conversation
            messages.append({"role": "assistant", "content": generated_assessment})

            # -----------------------------
            # 2. Reflect on the generated assessment
            # -----------------------------
            reflection_prompt = prompt_reflect.format(
                data_type="Assessment",
                ground_truth=ground_truth_assessment
            )
            messages.append({"role": "user", "content": reflection_prompt})

            gap_output = self.get_schema_followed_response(
                messages, 
                schema_set["gap_generation"]
            )

            if not gap_output:
                parsing_error += 1
                print(f"Error in generating Gap (Assessment) at index: {idx}")
                pbar.update(1)
                continue

            gap_assessment = gap_output["gap"]
            if gap_assessment:
                messages.append({"role": "assistant", "content": gap_assessment})
                # ---------------------
                # 3. Generate Lessons (round 1)
                # ---------------------
                messages.append({"role": "user", "content": prompt_lesson})
                lesson_output = self.get_schema_followed_response(
                    messages, 
                    schema_set["lesson_generation"]
                )
                if lesson_output:
                    self.general_memory.extend(lesson_output["general"])
                    self.specific_memory.extend(lesson_output["disease_specific"])
                    messages.append({"role": "assistant", "content": str(lesson_output)})
                else:
                    print(f"Error in generating Lessons (Assessment) at index: {idx}")
            else:
                print("Model is confident about its Assessment. No self-reflect needed.")
                messages.append({"role": "assistant", "content": "I am confident about my Assessment."})

            # ---------------------
            # 4. Generate Summary
            # ---------------------
            messages.append({"role": "user", "content": prompt_generate_summ})
            summ_output = self.get_schema_followed_response(
                messages, 
                schema_set["summary_generation"]
            )
            if not summ_output:
                parsing_error += 1
                print(f"Error in generating Summary at index: {idx}")
                pbar.update(1)
                continue

            generated_summary = summ_output["problems"]
            messages.append({"role": "assistant", "content": generated_summary})

            # ---------------------
            # 5. Reflect on the generated summary
            # ---------------------
            reflection_prompt = prompt_reflect.format(
                data_type="problem list",
                ground_truth=ground_truth_summary
            )
            messages.append({"role": "user", "content": reflection_prompt})

            gap_output = self.get_schema_followed_response(
                messages, 
                schema_set["gap_generation"]
            )
            if not gap_output:
                parsing_error += 1
                print(f"Error in generating Gap (Summary) at index: {idx}")
                pbar.update(1)
                continue

            gap_summary = gap_output["gap"]
            if gap_summary:
                messages.append({"role": "assistant", "content": gap_summary})
                # ---------------------
                # 6. Generate Lessons (round 2)
                # ---------------------
                messages.append({"role": "user", "content": prompt_lesson})
                lesson_output = self.get_schema_followed_response(
                    messages, 
                    schema_set["lesson_generation"]
                )
                if lesson_output:
                    self.general_memory.extend(lesson_output["general"])
                    self.specific_memory.extend(lesson_output["disease_specific"])
                else:
                    print(f"Error in generating Lessons (Summary) at index: {idx}")
            else:
                print("Model is confident about its Summary. No self-reflect needed.")

            # Save intermediate memory every 10 notes
            if (idx + 1) % 10 == 0:
                with open(f"{self.test_name}_memory_at_{idx+1}.pkl", "wb") as f:
                    pickle.dump((self.general_memory, self.specific_memory), f)
                self.refine_long_term_memory()

            pbar.update(1)

        pbar.close()
        # Final refinement and final save
        self.refine_long_term_memory()
        with open(f"{self.test_name}_memory_at_end.pkl", "wb") as f:
            pickle.dump((self.general_memory, self.specific_memory), f)

        return self.general_memory, self.specific_memory

    def refine_long_term_memory(self):
        """
        Refines the agent's long-term memory (both general and disease-specific) 
        by removing redundancies, contradictions, and confusion.
        """
        # ---------------------
        # General Memory Refinement
        # ---------------------
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": prompt_memory_refinement.format(
                    all_accumulated_memory=self.general_memory
                ),
            },
        ]
        refined_general_memory_output = self.get_schema_followed_response(
            messages, 
            schema_set["memory_refinement"]
        )

        if refined_general_memory_output:
            self.general_memory = refined_general_memory_output["refined_memory"]
        else:
            print("Error in refining general memory.")

        # ---------------------
        # Disease-Specific Memory Refinement
        # ---------------------
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": prompt_memory_refinement.format(
                    all_accumulated_memory=self.specific_memory
                ),
            },
        ]
        refined_specific_memory_output = self.get_schema_followed_response(
            messages, 
            schema_set["memory_refinement"]
        )

        if refined_specific_memory_output:
            self.specific_memory = refined_specific_memory_output["refined_memory"]
        else:
            print("Error in refining disease-specific memory.")


if __name__ == "__main__":
    # Example usage:
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1", timeout=120.0)
    df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/data/mergedBioNLP2023.csv")

    agent = Agent(
        client=client,
        model="meta-llama/Llama-3.3-70B-Instruct",
        test_name="ltm_build"
    )
    agent.build_memory(df)
