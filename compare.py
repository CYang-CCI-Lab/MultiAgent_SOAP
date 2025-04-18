# …everything above stays the same…

class Manager(LLMAgent):
    def __init__(
        self, 
        note: str, 
        hadm_id: str, 
        problem: str, 
        label: str, 
        n_specialists: Union[int, Literal["auto"]] = 'auto',
        static_agents: Optional[List[str]] = None,     # NEW: list of system prompts
        consensus_threshold: float = 0.8,
        max_consensus_attempts=3, 
        max_assignment_attempts=2,
    ):
        super().__init__(
            system_prompt=(
                "You are the manager of a multi‐agent diagnostic system. "
                "Your job is to coordinate sub‐agents to decide if the patient has the problem."
            )
        )
        self.note = note
        self.hadm_id = hadm_id
        self.problem = problem
        self.label = label

        self.static_agents = static_agents
        # if static_agents is provided, n_specialists is derived from its length
        if self.static_agents is not None:
            self.n_specialists = len(self.static_agents)
        else:
            self.n_specialists = n_specialists

        self.consensus_threshold = consensus_threshold
        self.max_consensus_attempts = max_consensus_attempts
        self.max_assignment_attempts = max_assignment_attempts
        self.assignment_attempts = 0
        self.consensus_attempts = 0

        # storage for panel info
        self.state_dict = {
            "note": note,
            "hadm_id": hadm_id,
            "problem": problem,
            "label": label,
            "panels": {},
            "final": {}
        }

    async def _assign_specialists(self):
        """Either create generic agents from static_agents, or fall back to LLM-driven specialties."""
        self.assignment_attempts += 1
        panel_key = f"panel_{self.assignment_attempts}"
        self.state_dict["panels"][panel_key] = {"sub_agents": {}}

        # If user supplied static_agents, use them directly:
        if self.static_agents is not None:
            for i, prompt in enumerate(self.static_agents, start=1):
                role_name = f"Agent_{i}"
                self.state_dict["panels"][panel_key]["sub_agents"][role_name] = {
                    "system_prompt": prompt,
                    "answer_history": {}
                }
            return self.state_dict

        # …otherwise, run your existing _assign_specialists logic for specialties…
        # (leave your original code here)
        # at the end, it should populate self.state_dict["panels"][panel_key]["sub_agents"]
        # mapping role_name → {"expertise": [...], "answer_history": {}}

        # e.g.:
        # return self.state_dict

    async def run(self):
        while self.assignment_attempts < self.max_assignment_attempts:
            panel_state = await self._assign_specialists()
            panel_key = f"panel_{self.assignment_attempts}"
            sub_agents_cfg = panel_state["panels"][panel_key]["sub_agents"]

            # Instantiate DynamicSpecialist-like agents, but now using system_prompt directly
            panel = []
            for role, cfg in sub_agents_cfg.items():
                # pass the *system prompt* (cfg["system_prompt"]) instead of building it from a specialty
                specialist = LLMAgent(system_prompt=cfg["system_prompt"])
                panel.append((role, specialist))

            # Step 1) each agent analyzes the note
            analyze = []
            for role, agent in panel:
                user_prompt = (
                    f"<SOAP>\n{self.note}\n</SOAP>\n\n"
                    f"Problem: {self.problem}\n\n"
                    "Please provide:\n"
                    "1) Your reasoning\n"
                    "2) Your choice: 'Yes' or 'No'"
                )
                analyze.append((role, agent.llm_call(user_prompt, temperature=0.1)))
            results = await asyncio.gather(*(task for _, task in analyze))

            for (role, _), parsed in zip(analyze, results):
                # assume safe_json_load
                self.state_dict["panels"][panel_key]["sub_agents"][role]["answer_history"]["round_1"] = parsed

            # check consensus
            self.consensus_attempts += 1
            choice = self._check_consensus(panel_key, round_id=1)
            if choice:
                self.state_dict["final"] = {"final_choice": choice, "reasoning": "Consensus reached"}
                return self.state_dict

            # you can add debate loops similarly…

            break  # for brevity, we stop here

        # fallback aggregation if no consensus
        # …
        return self.state_dict

    def _check_consensus(self, panel_key: str, round_id: int) -> Optional[str]:
        counts = {}
        sub_agents = self.state_dict["panels"][panel_key]["sub_agents"]
        threshold = math.ceil(self.n_specialists * self.consensus_threshold)
        for role, cfg in sub_agents.items():
            choice = cfg["answer_history"][f"round_{round_id}"]["choice"]
            counts[choice] = counts.get(choice, 0) + 1
            if counts[choice] >= threshold:
                return choice
        return None

# Usage in process_problem:

async def process_problem(df: pd.DataFrame, problem: str):
    static_prompts = [
        "You are a generic LLM with no specific medical specialty.",
        "You are another generic LLM consultant without a domain role."
        # add as many as you like...
    ]

    for idx, row in df.iterrows():
        manager = Manager(
            note=str(row["Subjective"]) + "\n" + str(row["Objective"]),
            hadm_id=row["File ID"],
            problem=problem,
            label=row["combined_summary"],
            static_agents=static_prompts,  # now we run with 2 generic agents
            consensus_threshold=0.8,
            max_consensus_attempts=2,
            max_assignment_attempts=1,
        )
        result = await manager.run()
        print(result)
