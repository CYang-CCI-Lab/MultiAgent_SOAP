async def run_generic(note, hadm_id, problem, label):
    mgr = Manager(note, hadm_id, problem, label,
                  n_generic_agents=5, consensus_threshold=0.8,
                  max_consensus_attempts=4, max_assignment_attempts=3)
    state = await mgr.run_generic_agents()
    return {
        "method": "generic_multi",
        "hadm_id": hadm_id,
        "choice": state["final"]["final_choice"],
        "reasoning": state["final"]["final_reasoning"],
        "raw_state": state,          # keep everything if you want
    }

async def run_dynamic(note, hadm_id, problem, label):
    mgr = Manager(note, hadm_id, problem, label,
                  n_specialists="auto", consensus_threshold=0.8,
                  max_consensus_attempts=4, max_assignment_attempts=3)
    state = await mgr.run_specialists()
    return {
        "method": "dynamic_multi",
        "hadm_id": hadm_id,
        "choice": state["final"]["final_choice"],
        "reasoning": state["final"]["final_reasoning"],
        "raw_state": state,
    }

async def run_baseline(note, hadm_id, problem, label):
    zs = BaselineZS()
    out = await zs.classify(note, problem)
    return {
        "method": "baseline_zs",
        "hadm_id": hadm_id,
        "choice": out["choice"],
        "reasoning": out["reasoning"],
        "raw_state": out,
    }

async def process_row(row, problem):
    note = f"{row['Subjective']}\n{row['Objective']}"
    hadm_id = row["File ID"]
    label = row["combined_summary"]

    tasks = [
        run_generic(note, hadm_id, problem, label),
        run_dynamic(note, hadm_id, problem, label),
        run_baseline(note, hadm_id, problem, label),
    ]
    return await asyncio.gather(*tasks)           # returns a list of 3 dicts

async def process_problem(df, problem):
    logger.info("Processing %s (%d rows).", problem, len(df))
    all_results = []

    for _, row in df.iterrows():
        all_results.extend(await process_row(row, problem))

    out_path = f"/…/results_{problem.replace(' ','_')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("[%s] saved to %s", problem, out_path)

async def main():
    df = pd.read_csv("/…/SOAP_all_problems.csv", lineterminator="\n")
    tasks = [asyncio.create_task(process_problem(df, p))
             for p in selected_problems]
    await asyncio.gather(*tasks)

# df = pd.read_json("results_acute_kidney_injury.json")
# df.groupby("method")["choice"].eq(df["label"]).mean()







####

# async def process_problem(df: pd.DataFrame, problem: str):
#     logger.info(f"Processing problem '{problem}' for {len(df)} rows.")
#     results = []

#     for idx, row in df.iterrows():
#         logger.info(f"[{problem}] Processing row index {idx}")

#         note_text = str(row["Subjective"]) + "\n" + str(row['Objective'])
#         hadm_id = row["File ID"]
#         label = row["combined_summary"]

#         manager = Manager(
#             note=note_text,
#             hadm_id=hadm_id,
#             problem=problem,
#             label=label,
#             # n_specialists='auto',  # or an integer
#             n_generic_agents=5,
#             consensus_threshold=0.8,
#             max_consensus_attempts=4,
#             max_assignment_attempts=3,
#         )

#         # Run the manager's workflow
#         result = await manager.run_generic_agents()
#         results.append(result)

#     # Save results for this problem
#     output_path = f"/home/yl3427/cylab/SOAP_MA/Output/SOAP/generic/generic_{problem.replace(' ', '_')}_new_temp.json"
#     with open(output_path, "w") as f:
#         json.dump(results, f, indent=4)
#     logger.info(f"[{problem}] Results saved to: {output_path}")

# async def main():
#     df_path = "/home/yl3427/cylab/SOAP_MA/Input/SOAP_all_problems.csv"
#     df = pd.read_csv(df_path, lineterminator='\n')
#     logger.info("Loaded dataframe with %d rows.", len(df))

#     # Create an asyncio Task for each problem
#     tasks = []
#     for problem in selected_problems:
#         tasks.append(asyncio.create_task(process_problem(df, problem)))

#     # Run them concurrently
#     await asyncio.gather(*tasks)

#     logger.info("All tasks completed.")
