# ──────────────── multi_agent_generic.py ────────────────
from typing import List, Literal, Dict, Any, Optional, Union
import json, math, asyncio, logging
from pydantic import BaseModel, Field

# import the core classes you already have
from multi_agent_original import LLMAgent, Manager, safe_json_load   # path = your file

logger = logging.getLogger(__name__)

class GenericAgent(LLMAgent):
    def __init__(self, agent_id: Union[str, int], state: dict):
        self.agent_id = str(agent_id)
        self.state = state
        self.state["static agents"][self.agent_id] = {} 

        self.round_id: int = 0
        self.schema = None
        system_prompt = "You are a collaborating diagnostic agent in a multi‑agent AI system designed to handle medical questions."

        super().__init__(system_prompt)

        logger.info(f"[GenericAgent '{self.agent_id}'] Initialized...")

    async def analyse_note(self, note: str, problem: str):
        self.round_id += 1

        class Response(BaseModel):
            reasoning: str = Field(..., description="Step-by-step reasoning leading to your choice.")
            choice: Literal["Yes", "No"] = Field(..., description=f"Your choice indicating whether the patient has {problem}.")
        self.schema = Response.model_json_schema()

        user_prompt = (
            " Read the patient note and decide whether the patient has the specified problem.\n"
            f"<<<PATIENT NOTE>>>\n{note}\n<<<END NOTE>>>\n\n"
            f"Question: Does this patient have {problem}?\n"
            "Please give your reasoning, then the choice ('Yes' or 'No')."
        )

        try:
            raw = await self.llm_call(
                user_prompt,
                temperature=0.1,
                guided_={"guided_json": self.schema}
            )
            parsed = safe_json_load(raw)
            self.append_message(raw)
            self.state["static agents"][self.agent_id][f"round_{self.round_id}"] = parsed
            return parsed
        except Exception as e:
            logger.error("[%s] analyse_note() failed: %s", self.agent_id, e)
            return None

    async def debate(self, panel_state: Dict[str, Any]): # panel_state : state["panel_1"]["Static Agents]
        self.round_id += 1
        peers = {
            agent: info[f"round_{self.round_id-1}"]
            for agent, info in panel_state.items()
            if agent != self.agent_id
        }

        user_prompt = (
                    "Here are your peers’ previous answers:\n"
                    f"{json.dumps(peers, indent=2)}\n\n"
                    "Please review your peers' reasoning and final choices. Based on their input and your own analysis, reconsider your initial assessment. "
                    "You may either keep your original conclusion or change it.\n\n"
                    "Please give your refined reasoning and final choice ('Yes' or 'No')."
                    )

        try:
            raw = await self.llm_call(
                user_prompt,
                temperature=0.3,
                guided_={"guided_json": self.schema}
            )
            parsed = safe_json_load(raw)
            self.append_message(raw)
            self.state["static agents"][self.agent_id][f"round_{self.round_id}"] = parsed
            return parsed
        except Exception as e:
            logger.error("[%s] debate() failed: %s", self.agent_id, e)
            return None


# ────────────────────────────────
# 2) Manager that *skips* specialty selection
# ────────────────────────────────
class ManagerGeneric(Manager):
    """
    Same high‑level workflow as your existing Manager, but:
    • skips the “choose specialties” LLM step
    • spawns a fixed number of GenericAgent instances
    """

    def __init__(
        self,
        note: str,
        hadm_id: str,
        problem: str,
        label: str,
        n_agents: int = 5,                       # <── number of sub‑agents you want
        consensus_threshold: float = 0.8,
        max_consensus_attempts: int = 3,
    ):
        super().__init__(
            note=note,
            hadm_id=hadm_id,
            problem=problem,
            label=label,
            n_specialists=n_agents,              # we *reuse* the old variable internally
            consensus_threshold=consensus_threshold,
            max_consensus_attempts=max_consensus_attempts,
            max_assignment_attempts=1,           # we assign only once
            static_specialists=None,
        )

    # ---------- override ONLY the specialist‑assignment ----------
    async def _assign_specialists(self):
        self.assignment_attempts += 1
        panel_key = f"panel_{self.assignment_attempts}"
        self.state_dict[panel_key] = {
            "Initially Identified Specialties": [],            # kept for compatibility
            "Collected Specialists": {}
        }

        # Simply create n generic agent labels
        agent_names = [f"Agent_{i+1}" for i in range(self.n_specialists)]
        for name in agent_names:
            self.state_dict[panel_key]["Collected Specialists"][name] = {
                "expertise": [],           # empty
                "answer_history": {}
            }

        logger.info("Assigned %d generic agents: %s", self.n_specialists, agent_names)
        return self.state_dict

    # ---------- override DynamicSpecialist → GenericAgent ----------
    async def run(self):
        # a *slim* re‑implementation of the part that instantiates sub‑agents
        await self._assign_specialists()
        panel_key = f"panel_{self.assignment_attempts}"
        panel = [
            GenericAgent(name)
            for name in self.state_dict[panel_key]["Collected Specialists"].keys()
        ]

        # — analysis round —
        analyse = [asyncio.create_task(agent.analyse_note(self.note, self.problem)) for agent in panel]
        await asyncio.gather(*analyse)
        self.consensus_attempts += 1
        choice = self._check_consensus(self.assignment_attempts, self.consensus_attempts)
        if choice:
            self.state_dict["final"] = {
                "final_choice": choice,
                "final_reasoning": "Consensus reached"
            }
            return self.state_dict

        # — debate rounds —
        while self.consensus_attempts < self.max_consensus_attempts:
            debate = [asyncio.create_task(agent.debate(
                self.state_dict[panel_key]["Collected Specialists"]
            )) for agent in panel]
            await asyncio.gather(*debate)
            self.consensus_attempts += 1
            choice = self._check_consensus(self.assignment_attempts, self.consensus_attempts)
            if choice:
                self.state_dict["final"] = {
                    "final_choice": choice,
                    "final_reasoning": "Consensus reached"
                }
                return self.state_dict

        # fall‑back aggregation (re‑use parent helper)
        agg = await self._aggregate()
        self.state_dict["final"] = {
            "final_choice": agg.get("aggregated_choice", "Error"),
            "final_reasoning": agg.get("aggregated_reasoning", "Aggregation failed")
        }
        return self.state_dict
# ─────────────────────────────────────────────────────────

