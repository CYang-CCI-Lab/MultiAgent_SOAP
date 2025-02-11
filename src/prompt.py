prompt_template_med42 = """
<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:
"""

system_instruction = """You are a medical expert in diagnostic reasoning."""

# ltm_builder = """
# You are provided with sections of a SOAP-formatted progress note for a patient diagnosed with {disease}, specifically the Subjective (S), Objective (O), and Assessment (A) sections.
# Identify patterns in how medical experts diagnose {disease} in the Assessment (A), based on the information provided in the Subjective (S) and Objective (O) sections. Formulate a list of rules or pieces of knowledge that can aid in diagnosing future patients with {disease}, using only the S and O sections.

# Progress Note of a Patient with {disease}:
# <Subjective>
# {subjective_section}
# </Subjective>
# <Objective>
# {objective_section}
# </Objective>
# <Assessment>
# {assessment_section}
# </Assessment>

# Structure your output as a list of rules, formatted as follows:
# ["Rule 1", "Rule 2", "Rule 3", ...]
# """

# ltm_builder_vllm = """
# You are provided with sections of a SOAP-formatted progress note for a patient ultimately diagnosed with {disease}, specifically the Subjective (S), Objective (O), and Assessment (A) sections.

# Your task is to identify patterns and formulate **general reasoning frameworks or rules** that clinicians could use to evaluate patients for potential {disease}, based on the information in the Subjective (S) and Objective (O) sections. These rules should help in assessing whether future patients might have {disease}, even when the diagnosis is not yet confirmed.

# Progress Note of a Patient with {disease}:
# <Subjective>
# {subjective_section}
# </Subjective>
# <Objective>
# {objective_section}
# </Objective>
# <Assessment>
# {assessment_section}
# </Assessment>

# Structure your output as a list of **general rules or reasoning steps**, formatted as follows:
# ["Rule 1: ...", "Rule 2: ...", "Rule 3: ...", ...]

# Guidelines:
# - Derive patterns or reasoning principles from the provided example.
# - Focus on how clinicians integrate information from S and O to arrive at potential assessments for {disease}.
# - Avoid specific details tied only to this patient; instead, abstract principles applicable to similar cases.
# - Consider how clinical reasoning could be applied in cases where {disease} is one of several possible diagnoses.

# These rules should provide a framework for evaluating and reasoning about future cases where {disease} is a diagnostic possibility.

# """


# ltm_refiner = """
# You are refining a list of diagnostic knowledge rules derived from analyzing patients' SOAP-formatted progress notes for {disease}. These rules are intended to assist in correctly diagnosing {disease} based on information from the Subjective (S) and Objective (O) sections of patient progress notes.

# Your Task:
# - Review each rule in the list carefully and consolidate similar or overlapping rules into single, more general rules.
# - Rewrite overly specific rules to capture underlying principles applicable to a broader range of cases.
# - Ensure that the refined list maintains the usefulness of the original rules while being more concise and general.
# - Do not add any new rules; only refine the existing ones.
# - Structure your output as a list of rules, formatted as ["Refined Rule 1", "Refined Rule 2", "Refined Rule 3", ...].


# Original List of Rules:
# {original_rules}

# Your Refined List of Rules:
# """

# ltm_refiner_vllm = """
# You are refining a list of diagnostic knowledge rules derived from analyzing patients' SOAP-formatted progress notes for {disease}. These rules aim to assist clinicians in reasoning through potential diagnoses of {disease} based on information from the Subjective (S) and Objective (O) sections.

# Your Task:
# 1. **Consolidate Overlapping Rules**: Combine similar or overlapping rules into single, broader rules that capture their shared essence.
# 2. **Generalize Overly Specific Rules**: Rewrite any rules that are too specific to individual cases so they reflect general diagnostic principles or reasoning patterns.
# 3. **Ensure Usefulness and Clarity**: Retain the diagnostic utility of the rules while making the list more concise, clear, and broadly applicable.
# 4. **Do Not Add New Rules**: Only refine or consolidate the rules provided; no additional rules should be introduced.
# 5. **List Format**: Present your refined rules in the following format:
#    ["Refined Rule 1", "Refined Rule 2", "Refined Rule 3", ...]

# Original List of Rules:
# {original_rules}

# Your Refined List of Rules:
# """

ltm_test = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective and Objective sections.
You have access to a list of diagnostic knowledge rules specifically related to diagnosing {disease}.

Your task is to determine if the patient has {disease} based on the provided rules and patient information. You must provide a clear Yes/No response.

Output format: Yes/No

Knowledge Rules:
{list_of_rules}

Patient Information:
<Subjective>
{subjective_section}
</Subjective> 
<Objective>
{objective_section}
</Objective>

"""


without_ltm_test = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective and Objective sections.

Your task is to determine if the patient has {disease} based on the provided patient information. You must provide a clear Yes/No response.

Output format: Yes/No

Patient Information:
<Subjective>
{subjective_section}
</Subjective> 
<Objective>
{objective_section}
</Objective>

"""

cot1_test = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective and Objective sections.

Your task is to write the Assessment (A) section by analyzing the provided information in the Subjective and Objective sections. Ensure the Assessment is concise, accurate, and clinically relevant.

Patient Information:
<Subjective>
{subjective_section}
</Subjective>
<Objective>
{objective_section}
</Objective>

"""


ltm_cot2_test = """
You now have the Assessment (A) section that you wrote, as well as a list of diagnostic knowledge rules specifically related to diagnosing {disease}.

Your task is to determine if the patient has {disease} by comparing the Assessment against the provided diagnostic rules. You must provide a clear Yes/No response.

Output format: Yes/No

Diagnostic Knowledge Rules:
{list_of_rules}

"""

without_ltm_cot2_test = """
You now have the Assessment (A) section that you wrote.

Your task is to determine if the patient has {disease}, based on the Assessment. You must provide a clear "Yes" or "No" response.

Output format: Yes/No

"""


soap_soa = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective, Objective, and Assessment sections. Your task is to generate a summary that lists the patient's medical problems and diagnoses, including both direct and indirect problems (a past medical problem or consequence from the primary diagnosis). Present your response as a list of diagnoses.

Patient Information:
<Subjective>
{subjective_section}
</Subjective> 

<Objective>
{objective_section}
</Objective>

<Assessment>
{assessment_section}
</Assessment>

"""
