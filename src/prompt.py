####### SOAP Prompts: S, O, A -> Summary #######
soap_soa_zs = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective, Objective, and Assessment sections. Your task is to generate a summary that lists the patient's medical problems and diagnoses, including both direct and indirect problems (a past medical problem or consequence from the primary diagnosis). Present your response as a concatenated list of diagnoses separated by semicolons without any additional text or formatting.

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

soap_soa_os = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective, Objective, and Assessment sections. Your task is to generate a summary that lists the patient's medical problems and diagnoses, including both direct and indirect problems (a past medical problem or consequence from the primary diagnosis). Use the example provided to guide your response. Present your response as a concatenated list of diagnoses separated by semicolons without any additional text or formatting.

-----
Example

Patient Information:
<Subjective>
TITLE: Chief Complaint: 24 Hour Events: Allergies: No Known Drug Allergies
</Subjective>

<Objective>
Last dose of Antibiotics: Infusions: Other ICU medications: Other medications: Changes to medical and family history: Review of systems is unchanged from admission except as noted below Review of systems: Flowsheet Data as of   07:07 AM Vital signs Hemodynamic monitoring Fluid balance 24 hours Since 12 AM Tmax: 36.6 C (97.9 Tcurrent: 36.3 C (97.4 HR: 54 (42 - 76) bpm BP: 142/54(75) (114/32(56) - 147/76(90)) mmHg RR: 17 (12 - 19) insp/min SpO2: 99% Heart rhythm: SR (Sinus Rhythm) Height: 76 Inch Total In: 900 mL PO: 900 mL TF: IVF: Blood products: Total out: 680 mL 980 mL Urine: 680 mL 980 mL NG: Stool: Drains: Balance: -680 mL -80 mL Respiratory support SpO2: 99% ABG: ///27/ Physical Examination GENERAL: Alert, interactive, comfortable, NAD. HEENT: Enlarged 1cm (approx) uvula with erythema and swelling of left side of soft palate/arch. CARDIAC: RRR, normal S1, S2. No m/r/g. LUNGS: Resp unlabored, no accessory muscle use. CTAB, no crackles, wheezes or rhonchi. ABDOMEN: Soft, NTND. EXTREMITIES: No c/c/e. Labs / Radiology 240 K/uL 13.9 g/dL 150 mg/dL 1.0 mg/dL 27 mEq/L 4.5 mEq/L 13 mg/dL 104 mEq/L 139 mEq/L 42.3 % 6.1 K/uL [image002.jpg]   04:01 AM WBC 6.1 Hct 42.3 Plt 240 Cr 1.0 Glucose 150 Other labs: Differential-Neuts:83.6 %, Lymph:10.6 %, Mono:5.1 %, Eos:0.3 %, Ca++:9.8 mg/dL, Mg++:1.8 mg/dL, PO4:3.7 mg/dL
</Objective>

<Assessment>
Mr. [**Known lastname 8748**] is a 19 year old gentleman with history of AVNRT s/p nodal\n   ablation on [**2182-11-7**] with post procedural swelling of uvula.
</Assessment>

Summary:
Uvula swelling; AVNRT
-----

Now, please perform the same task on the following patient information:

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


####### SOAP Prompts: A -> Summary #######
soap_a_zs = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Assessment section. Your task is to generate a summary that lists the patient's medical problems and diagnoses, including both direct and indirect problems (a past medical problem or consequence from the primary diagnosis). Present your response as a concatenated list of diagnoses separated by semicolons without any additional text or formatting.

Patient Information:
<Assessment>
{assessment_section}
</Assessment>

"""

soap_a_os = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Assessment section. Your task is to generate a summary that lists the patient's medical problems and diagnoses, including both direct and indirect problems (a past medical problem or consequence from the primary diagnosis). Use the example provided to guide your response. Present your response as a concatenated list of diagnoses separated by semicolons without any additional text or formatting.

-----
Example

Patient Information:
<Assessment>
Mr. [**Known lastname 8748**] is a 19 year old gentleman with history of AVNRT s/p nodal\n   ablation on [**2182-11-7**] with post procedural swelling of uvula.
</Assessment>

Summary:
Uvula swelling; AVNRT
-----

Now, please perform the same task on the following patient information:

Patient Information:
<Assessment>
{assessment_section}
</Assessment>

"""

####### SOAP Prompts: S, O -> Summary #######
soap_so_zs = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective and Objective sections. Your task is to generate a summary that lists the patient's medical problems and diagnoses, including both direct and indirect problems (a past medical problem or consequence from the primary diagnosis). Present your response as a concatenated list of diagnoses separated by semicolons without any additional text or formatting.

Patient Information:
<Subjective>
{subjective_section}
</Subjective> 

<Objective>
{objective_section}
</Objective>

"""

soap_so_os = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective and Objective sections. Your task is to generate a summary that lists the patient's medical problems and diagnoses, including both direct and indirect problems (a past medical problem or consequence from the primary diagnosis). Use the example provided to guide your response. Present your response as a concatenated list of diagnoses separated by semicolons without any additional text or formatting.

-----
Example

Patient Information:
<Subjective>
TITLE: Chief Complaint: 24 Hour Events: Allergies: No Known Drug Allergies
</Subjective>

<Objective>
Last dose of Antibiotics: Infusions: Other ICU medications: Other medications: Changes to medical and family history: Review of systems is unchanged from admission except as noted below Review of systems: Flowsheet Data as of   07:07 AM Vital signs Hemodynamic monitoring Fluid balance 24 hours Since 12 AM Tmax: 36.6 C (97.9 Tcurrent: 36.3 C (97.4 HR: 54 (42 - 76) bpm BP: 142/54(75) (114/32(56) - 147/76(90)) mmHg RR: 17 (12 - 19) insp/min SpO2: 99% Heart rhythm: SR (Sinus Rhythm) Height: 76 Inch Total In: 900 mL PO: 900 mL TF: IVF: Blood products: Total out: 680 mL 980 mL Urine: 680 mL 980 mL NG: Stool: Drains: Balance: -680 mL -80 mL Respiratory support SpO2: 99% ABG: ///27/ Physical Examination GENERAL: Alert, interactive, comfortable, NAD. HEENT: Enlarged 1cm (approx) uvula with erythema and swelling of left side of soft palate/arch. CARDIAC: RRR, normal S1, S2. No m/r/g. LUNGS: Resp unlabored, no accessory muscle use. CTAB, no crackles, wheezes or rhonchi. ABDOMEN: Soft, NTND. EXTREMITIES: No c/c/e. Labs / Radiology 240 K/uL 13.9 g/dL 150 mg/dL 1.0 mg/dL 27 mEq/L 4.5 mEq/L 13 mg/dL 104 mEq/L 139 mEq/L 42.3 % 6.1 K/uL [image002.jpg]   04:01 AM WBC 6.1 Hct 42.3 Plt 240 Cr 1.0 Glucose 150 Other labs: Differential-Neuts:83.6 %, Lymph:10.6 %, Mono:5.1 %, Eos:0.3 %, Ca++:9.8 mg/dL, Mg++:1.8 mg/dL, PO4:3.7 mg/dL
</Objective>

Summary:
Uvula swelling; AVNRT
-----

Now, please perform the same task on the following patient information:

Patient Information:
<Subjective>
{subjective_section}
</Subjective> 

<Objective>
{objective_section}
</Objective>

"""

####### SOAP Prompts: S, A -> Summary #######
soap_sa_zs = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective and Assessment sections. Your task is to generate a summary that lists the patient's medical problems and diagnoses, including both direct and indirect problems (a past medical problem or consequence from the primary diagnosis). Present your response as a concatenated list of diagnoses separated by semicolons without any additional text or formatting.

Patient Information:
<Subjective>
{subjective_section}
</Subjective> 

<Assessment>
{assessment_section}
</Assessment>

"""

soap_sa_os = """
You are provided with a patient's medical information from a progress note formatted in the SOAP structure, containing only the Subjective and Assessment sections. Your task is to generate a summary that lists the patient's medical problems and diagnoses, including both direct and indirect problems (a past medical problem or consequence from the primary diagnosis). Use the example provided to guide your response. Present your response as a concatenated list of diagnoses separated by semicolons without any additional text or formatting.

-----
Example

Patient Information:
<Subjective>
TITLE: Chief Complaint: 24 Hour Events: Allergies: No Known Drug Allergies
</Subjective>

<Assessment>
Mr. [**Known lastname 8748**] is a 19 year old gentleman with history of AVNRT s/p nodal\n   ablation on [**2182-11-7**] with post procedural swelling of uvula.
</Assessment>

Summary:
Uvula swelling; AVNRT
-----

Now, please perform the same task on the following patient information:

Patient Information:
<Subjective>
{subjective_section}
</Subjective> 

<Assessment>
{assessment_section}
</Assessment>

"""