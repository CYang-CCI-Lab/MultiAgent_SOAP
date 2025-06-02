#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  Multi‑Agent Medical‑QA System
#  ---------------------------------
#  A direct adaptation of the SOAP‑note yes/no diagnostic framework.  The core
#  multi‑agent logic is unchanged, but the pipeline now accepts *multiple‑choice
#  medical questions* and returns the single best answer (A–E).
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio, json, logging, math
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, create_model

# ──────────────────────────────────────────────────────────────────────────────
#  External helpers (unchanged from the original codebase)
#     – `count_llama_tokens`   : estimate Llama‑3 token usage for a message list
#     – `safe_json_load`       : robust `json.loads` that returns `None` on failure
# ──────────────────────────────────────────────────────────────────────────────
from utils import count_llama_tokens, safe_json_load          # Local utilities
from rag_retrieve import create_documents, hybrid_query       # Kept for parity

# ──────────────────────────────────────────────────────────────────────────────
#  Globals & logging
# ──────────────────────────────────────────────────────────────────────────────
LLAMA3_70B_MAX_TOKENS = 22000
CHOICES               = ["A", "B", "C", "D", "E"]

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Pydantic response schema
# ──────────────────────────────────────────────────────────────────────────────
class ResponseMC(BaseModel):
    """Structured answer returned by every agent."""
    reasoning: str                       = Field(..., description="Step‑by‑step reasoning.")
    choice:    Literal["A","B","C","D","E"] = Field(..., description="answer.")


# ──────────────────────────────────────────────────────────────────────────────
#  Core LLM wrapper
# ──────────────────────────────────────────────────────────────────────────────
class LLMAgent:
    """
    Thin convenience wrapper around the async OpenAI client that handles
    history management, summarization, and (optionally) tool calls.
    """
    def __init__(
        self,
        system_prompt: str,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        client          = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy"),
        max_tokens: int = LLAMA3_70B_MAX_TOKENS,
        summarization_threshold: float = 0.70
    ):
        self.model_name  = model_name
        self.client      = client
        self.messages    = [{"role": "system", "content": system_prompt}]
        self.max_tokens  = max_tokens
        self.token_thres = int(max_tokens * summarization_threshold)

        logger.info(
            "[%s] initialised – hard limit %d tks, summarize ≥ %d tks",
            self.__class__.__name__, max_tokens, self.token_thres
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Summarization helpers (unchanged apart from cosmetics)
    # ──────────────────────────────────────────────────────────────────────
    async def _summarize_once(self, text: str, max_chars: Optional[int] = None) -> Optional[str]:
        limit = f"Do **not** exceed {max_chars} characters." if max_chars else "Do **not** exceed 1000 words."
        prompt = (
            "Summarize the following message concisely, *preserving every key fact and reasoning step.*\n"
            f"{limit}\n\n<<<MESSAGE_START>>>\n{text}\n<<<MESSAGE_END>>>"
        )
        try:
            rsp = await self.client.chat.completions.create(
                model       = self.model_name,
                messages    = [
                    {"role": "system", "content": "You are a summarization assistant."},
                    {"role": "user",   "content": prompt}
                ],
                temperature = 0.1,
                max_tokens  = 1500
            )
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            return None

    async def _summarize_history(
        self,
        *,
        inplace: bool = True,
        message: str = "",
        max_chars: Optional[int] = None
    ) -> Union[bool, str]:
        """
        If *inplace* is True the longest user/assistant messages in `self.messages`
        are replaced by summaries until the running token count drops below the
        threshold.  If *inplace* is False, `message` is summarized externally.
        """
        if inplace:
            if len(self.messages) < 3:
                return False
            candidates: set[int] = set()
            while count_llama_tokens(self.messages) >= self.token_thres:
                idx_len = sorted(
                    ((i, count_llama_tokens([m]))
                     for i, m in enumerate(self.messages)
                     if m["role"] != "system" and i not in candidates),
                    key=lambda t: t[1], reverse=True
                )
                if not idx_len:
                    return False
                i = idx_len[0][0]
                summary = await self._summarize_once(self.messages[i]["content"], max_chars)
                if summary is None or count_llama_tokens([{"role": "assistant", "content": summary}]) >= count_llama_tokens([self.messages[i]]):
                    candidates.add(i)
                    continue
                self.messages[i]["content"] = f"[Summary] {summary}"
                logger.info("→ summarized message %d  (total %d tks)", i, count_llama_tokens(self.messages))
            return True

        # external mode
        if not message:
            return False
        summary = await self._summarize_once(message, max_chars)
        if summary is None:
            return False
        while count_llama_tokens(summary) >= self.token_thres or (max_chars and len(summary) > max_chars):
            summary = await self._summarize_once(summary, max_chars)
            if summary is None:
                return False
        return summary

    # ──────────────────────────────────────────────────────────────────────
    #  Thin convenience wrapper around the chat endpoint
    # ──────────────────────────────────────────────────────────────────────
    async def llm_call(
        self,
        user_prompt: str,
        *,
        temperature: float = 0.5,
        guided_: Optional[dict] = None
    ) -> str:
        if count_llama_tokens(self.messages + [{"role": "user", "content": user_prompt}]) > self.token_thres:
            ok = await self._summarize_history()
            if not ok and count_llama_tokens(self.messages) > self.max_tokens:
                raise RuntimeError("Context length still exceeds hard limit after summarization.")
        self.messages.append({"role": "user", "content": user_prompt})

        params: Dict[str, Any] = {
            "model":       self.model_name,
            "messages":    self.messages,
            "temperature": temperature,
        }
        if guided_:
            params["extra_body"] = guided_

        rsp = await self.client.chat.completions.create(**params)
        return rsp.choices[0].message.content

    # small helper used by child classes
    def append_message(self, content: Any, *, role: str = "assistant") -> None:
        self.messages.append({"role": role, "content": str(content)})


# ──────────────────────────────────────────────────────────────────────────────
#  Concrete agent types
# ──────────────────────────────────────────────────────────────────────────────
class BaselineZS(LLMAgent):
    def __init__(self):
        super().__init__("You are a clinical reasoning assistant.")
        self.schema = ResponseMC.model_json_schema()

    async def analyse_question(self, q_text: str) -> Optional[dict]:
        prompt = (
            "Multiple‑choice medical question (A–E):\n\n"
            f"{q_text}\n\n"
            "Provide your reasoning, then your choice."
        )
        raw = await self.llm_call(prompt, guided_={"guided_json": self.schema})
        return safe_json_load(raw)


class GenericAgent(LLMAgent):
    """
    A peer‑to‑peer diagnostic agent that participates in debate rounds.
    """
    def __init__(self, agent_id: str, state: dict):
        self.agent_id = str(agent_id)
        self.state    = state
        self.state.setdefault("generic_agents", {})[self.agent_id] = {}
        self.round    = 0
        self.schema   = ResponseMC.model_json_schema()

        super().__init__("You are a collaborating clinical agent in a multi‑agent AI system.")

    async def analyse_question(self, q_text: str) -> dict:
        self.round += 1
        prompt = (
            "Multiple‑choice medical question (A–E):\n\n"
            f"{q_text}\n\n"
            "Provide your reasoning, then your choice."
        )
        raw = await self.llm_call(prompt, guided_={"guided_json": self.schema})
        parsed = safe_json_load(raw)
        self.append_message(raw)
        self.state["generic_agents"][self.agent_id][f"round_{self.round}"] = parsed
        return parsed

    async def debate(self) -> dict:
        self.round += 1

        # gather peers’ most recent answers
        peers: Dict[str, Any] = {}
        for aid, hist in self.state["generic_agents"].items():
            if aid == self.agent_id:             # skip self
                continue
            tag = f"round_{self.round - 1}"
            if tag in hist:
                peers[aid] = hist[tag]
        for p in [k for k in self.state if k.startswith("panel_")]:
            for role, info in self.state[p]["Collected Specialists"].items():
                tag = f"round_{self.round - 1}"
                if tag in info["answer_history"]:
                    peers[role] = info["answer_history"][tag]

        prompt = (
            "Here are your peers’ previous answers:\n"
            f"{json.dumps(peers, indent=2)}\n\n"
            "Review their reasoning against yours. Considering their input and your own "
            "analysis, decide whether to revise your answer.\n\n"
            "Return your updated reasoning and final choice letter (A–E)."
        )
        raw    = await self.llm_call(prompt, guided_={"guided_json": self.schema})
        parsed = safe_json_load(raw)
        self.append_message(raw)
        self.state["generic_agents"][self.agent_id][f"round_{self.round}"] = parsed
        return parsed


class DynamicSpecialist(LLMAgent):
    """
    Role‑playing specialist created dynamically by the Manager.
    """
    def __init__(self, specialty: str, panel_id: int, state: dict):
        self.specialty   = specialty
        self.panel_id    = panel_id
        self.state       = state
        rec              = state[f"panel_{panel_id}"]["Collected Specialists"][specialty]
        self.expertise   = rec["expertise"]
        self.answer_hist = rec["answer_history"]
        self.round       = 0
        self.schema      = ResponseMC.model_json_schema()

        super().__init__(
            f"You are a **{specialty}**.  Relevant expertise:\n{self.expertise}\n"
            "Apply this knowledge to solve the question."
        )

    async def analyse_question(self, q_text: str) -> dict:
        self.round += 1
        prompt = (
            "Multiple‑choice medical question (A–E):\n\n"
            f"{q_text}\n\n"
            "Provide your reasoning, then your choice."
        )
        raw = await self.llm_call(prompt, guided_={"guided_json": self.schema})
        parsed = safe_json_load(raw)
        self.append_message(raw)
        self.answer_hist[f"round_{self.round}"] = parsed
        return parsed

    async def debate(self) -> dict:
        self.round += 1
        peers: Dict[str, Any] = {}

        # specialists in the same panel
        for role, info in self.state[f"panel_{self.panel_id}"]["Collected Specialists"].items():
            if role == self.specialty:
                continue
            tag = f"round_{self.round - 1}"
            if tag in info["answer_history"]:
                peers[role] = info["answer_history"][tag]

        # generic agents
        for gid, hist in self.state.get("generic_agents", {}).items():
            tag = f"round_{self.round - 1}"
            if tag in hist:
                peers[gid] = hist[tag]

        prompt = (
            "Here are your peers’ previous answers:\n"
            f"{json.dumps(peers, indent=2)}\n\n"
            "Review their reasoning against yours. Considering their input and your own "
            "analysis, decide whether to revise your answer.\n\n"
            "Return your updated reasoning and final choice letter (A–E)."
        )
        raw    = await self.llm_call(prompt, guided_={"guided_json": self.schema})
        parsed = safe_json_load(raw)
        self.append_message(raw)
        self.answer_hist[f"round_{self.round}"] = parsed
        return parsed


# ──────────────────────────────────────────────────────────────────────────────
#  Manager – coordinates everything
# ──────────────────────────────────────────────────────────────────────────────
class Manager(LLMAgent):
    def __init__(
        self,
        q_text: str,
        q_id: str,
        label: str,
        n_specialists: Union[int, Literal["auto"]] = "auto",
        n_generic: int = 0,
        static_specialists: Optional[List[str]] = None,
        consensus_thresh: float = 0.8,
        max_consensus_rounds: int = 3,
        max_assignment_attempts: int = 2,
        per_agent_summary_chars: int = 300,
    ):
        super().__init__(
            "You are the manager of a multi‑agent clinical reasoning system. "
            "Co‑ordinate the sub‑agents to reach a final answer."
        )

        self.q_text         = q_text
        self.q_id           = q_id
        self.label          = label

        self.n_specialists  = n_specialists
        self.n_generic      = n_generic
        self.static_specs   = static_specialists or []

        self.cons_thresh    = consensus_thresh
        self.max_cons_round = max_consensus_rounds
        self.max_assign     = max_assignment_attempts
        self.per_agent_chars= per_agent_summary_chars

        self.assign_attempts = 0
        self.state: Dict[str, Any] = {
            "question": q_text,
            "q_id":     q_id,
            "label":    label,
            "generic_agents": {},
            "final":    {}
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Specialist selection
    # ──────────────────────────────────────────────────────────────────────
    async def _assign_specialists(self) -> None:
        self.assign_attempts += 1
        pid = self.assign_attempts
        self.state[f"panel_{pid}"] = {
            "Initially Identified Specialties": [],
            "Collected Specialists": {}
        }

        # (1) decide which specialties we want
        specialties: List[str] = list(self.static_specs)
        additional_needed: Union[int, Literal["auto"]]
        if self.n_specialists == "auto":
            additional_needed = "auto"
        else:
            additional_needed = max(self.n_specialists - len(self.static_specs), 0)

        if additional_needed == 0 and not specialties:
            return                                          # nothing to add

        if additional_needed:
            ask = (
                f"Below is the **medical question** we need to solve:\n\n{self.q_text}\n\n"
                f"Already chosen specialties: {self.static_specs or '(none)'}.\n"
                "Please list any **additional medical specialties** whose expertise would help answer this question."
            )
            if additional_needed == "auto":
                class SpecialtyList(BaseModel):
                    specialties: List[str]
                raw = await self.llm_call(ask, guided_={"guided_json": SpecialtyList.model_json_schema()})
                specialties.extend(safe_json_load(raw)["specialties"])
            else:
                fld = {f"specialty_{i+1}": (str, ...) for i in range(additional_needed)}
                Req = create_model("RequestedSpecialties", **fld)
                raw = await self.llm_call(ask, guided_={"guided_json": Req.model_json_schema()})
                specialties.extend([safe_json_load(raw)[f"specialty_{i+1}"] for i in range(additional_needed)])

        self.state[f"panel_{pid}"]["Initially Identified Specialties"] = specialties

        # (2) flesh them out
        ask = (
            f"We will run a multi-agent debate to answer one question:\n"
            f"{self.q_text}\n\n"
            "The required specialties are:\n"
            f"{specialties}\n"
            "For each listed specialty return an object with keys:\n"
            "  • `specialist` – full job title for the agent to play\n"
            "  • `expertise`  – 1‑3 short phrases explaining the specialist’s expertise" 
        )
        class Specialist(BaseModel):
            specialist: str
            expertise:  List[str]
        fld  = {f"specialist_{i+1}": (Specialist, ...) for i in range(len(specialties))}
        Out  = create_model("SpecialistPanel", **fld)
        raw  = await self.llm_call(ask, guided_={"guided_json": Out.model_json_schema()})

        panel_json = safe_json_load(raw)
        self.append_message(raw)                           # keep full JSON record

        for obj in panel_json.values():
            self.state[f"panel_{pid}"]["Collected Specialists"][obj["specialist"]] = {
                "expertise":       obj["expertise"],
                "answer_history":  {}
            }

    # ──────────────────────────────────────────────────────────────────────
    #  Helper: panel summary in natural language (for re‑assignment phase)
    # ──────────────────────────────────────────────────────────────────────
    async def _panel_summary(self, pid: int) -> str:
        # find the most recent round answered by *any* agent
        rounds: List[int] = []
        for info in self.state[f"panel_{pid}"]["Collected Specialists"].values():
            rounds.extend(int(tag.split('_')[1]) for tag in info["answer_history"])
        for hist in self.state.get("generic_agents", {}).values():
            rounds.extend(int(tag.split('_')[1]) for tag in hist)
        if not rounds:
            return "(no answers recorded)"
        last = max(rounds)

        # collate reasoning → compress
        blobs: List[tuple[str,str,str]] = []
        for role, info in self.state[f"panel_{pid}"]["Collected Specialists"].items():
            tag = f"round_{last}"
            if tag in info["answer_history"]:
                ans = info["answer_history"][tag]
                blobs.append((role, ans["choice"], ans["reasoning"]))
        for gid, hist in self.state.get("generic_agents", {}).items():
            tag = f"round_{last}"
            if tag in hist:
                ans = hist[tag]
                blobs.append((gid, ans["choice"], ans["reasoning"]))

        async def _compress(text: str) -> str:
            out = await self._summarize_history(inplace=False, message=text, max_chars=self.per_agent_chars)
            return out if isinstance(out, str) else text.strip()

        mini = await asyncio.gather(*(_compress(r) for _, _, r in blobs))
        return "\n".join(f"- **{name}** → {ch} | {short}" for (name, ch, _), short in zip(blobs, mini))

    # ──────────────────────────────────────────────────────────────────────
    #  Helper: consensus check
    # ──────────────────────────────────────────────────────────────────────
    def _check_consensus(self, pid: int, rid: int, total: int) -> Optional[str]:
        tally: Dict[str, int] = {}
        for role, info in self.state[f"panel_{pid}"]["Collected Specialists"].items():
            tag = f"round_{rid}"
            if tag in info["answer_history"]:
                c = info["answer_history"][tag]["choice"]
                tally[c] = tally.get(c, 0) + 1

        for gid, hist in self.state.get("generic_agents", {}).items():
            tag = f"round_{rid}"
            if tag in hist:
                c = hist[tag]["choice"]
                tally[c] = tally.get(c, 0) + 1

        for choice, n in tally.items():
            if n >= math.ceil(total * self.cons_thresh):
                return choice
        return None

    # ──────────────────────────────────────────────────────────────────────
    #  Main entry
    # ──────────────────────────────────────────────────────────────────────
    async def run_hybrid(self) -> dict:
        crashed: List[str] = []

        while self.assign_attempts < self.max_assign:
            if self.assign_attempts:              # re‑assignment
                prev = self.assign_attempts
                summary = await self._panel_summary(prev)
                self.append_message(
                    "The previous specialist panel failed to reach a consensus.\n\n"
                    f"**Final positions from the panel:**\n{summary}\n\n"
                    "Please assemble a *fresh* set of specialties to resolve the disagreement.",
                    role="system"
                )

            await self._assign_specialists()
            pid = self.assign_attempts

            specs = [
                DynamicSpecialist(role, pid, self.state)
                for role in self.state[f"panel_{pid}"]["Collected Specialists"]
            ]
            gens  = [GenericAgent(f"generic_{i+1}", self.state) for i in range(self.n_generic)]
            agents: List[Union[DynamicSpecialist, GenericAgent]] = specs + gens
            crashed.clear()

            # ─── Round 1
            results = await asyncio.gather(
                *(ag.analyse_question(self.q_text) for ag in agents),
                return_exceptions=True
            )
            agents = [ag for ag, res in zip(agents, results) if not isinstance(res, Exception)]
            crashed.extend(
                ag.agent_id if hasattr(ag, "agent_id") else ag.specialty
                for ag, res in zip(specs+gens, results) if isinstance(res, Exception)
            )
            if not agents:
                raise RuntimeError(f"All agents crashed for QID {self.q_id}")

            choice = self._check_consensus(pid, 1, len(agents))
            if choice:
                self.state["final"] = {"final_reasoning": "Consensus reached", "final_choice": choice}
                self.state["meta"]  = {"crashed_agents": crashed, "active_agents": len(agents), "round": 1}
                return self.state

            # ─── Debate rounds
            for r in range(2, self.max_cons_round + 1):
                results = await asyncio.gather(*(ag.debate() for ag in agents), return_exceptions=True)
                agents  = [ag for ag, res in zip(agents, results) if not isinstance(res, Exception)]
                crashed.extend(
                    ag.agent_id if hasattr(ag, "agent_id") else ag.specialty
                    for ag, res in zip(agents, results) if isinstance(res, Exception)
                )
                if not agents:
                    raise RuntimeError(f"No agents survived round {r} for QID {self.q_id}")

                choice = self._check_consensus(pid, r, len(agents))
                if choice:
                    self.state["final"] = {"final_reasoning": "Consensus reached", "final_choice": choice}
                    self.state["meta"]  = {"crashed_agents": crashed, "active_agents": len(agents), "round": r}
                    return self.state

        # ───  Fallback: aggregate with a single LLM call
        prompt = (
            "The specialists failed to reach consensus. Below is the entire "
            "conversation history:\n"
            f"{json.dumps({k: v for k, v in self.state.items() if k not in ('final', 'label')}, indent=2)}\n\n"
            "Please:\n"
            " 1) Provide a concise explanation of how you chose the final answer.\n"
            " 2) Output the single best‑supported choice (A–E).\n"
        )
        class AggOut(BaseModel):
            final_reasoning: str                              = Field(..., description="Concise explanation.")
            final_choice:   Literal["A","B","C","D","E"] = Field(..., description="Final choice.")
        raw = await self.llm_call(prompt, guided_={"guided_json": AggOut.model_json_schema()})
        self.state["final"] = safe_json_load(raw) or {"final_choice": "ERROR", "final_reasoning": "LLM failure"}
        self.state["meta"]  = {"crashed_agents": crashed, "active_agents": len(agents), "round": None}
        return self.state


# ──────────────────────────────────────────────────────────────────────────────
#  Convenience wrappers (same five modes as before)
# ──────────────────────────────────────────────────────────────────────────────
async def run_baseline(q_text: str, q_id: str, label: str) -> dict:
    agent = BaselineZS()
    out   = await agent.analyse_question(q_text)
    return {
        "method":   "baseline_zs",
        "q_id":     q_id,
        "label":    label,
        "choice":   (out or {}).get("choice", "ERROR"),
        "reasoning":(out or {}).get("reasoning", "failure"),
        "raw_state": out,
    }


async def run_generic(q_text: str, q_id: str, label: str) -> dict:
    mgr = Manager(q_text, q_id, label, n_specialists=0, n_generic=5)
    st  = await mgr.run_hybrid()
    return {
        "method":    "generic",
        "q_id":      q_id,
        "label":     label,
        "choice":    st["final"]["final_choice"],
        "reasoning": st["final"]["final_reasoning"],
        "raw_state": st,
    }


async def run_dynamic(q_text: str, q_id: str, label: str) -> dict:
    mgr = Manager(q_text, q_id, label, n_specialists=5, n_generic=0)
    st  = await mgr.run_hybrid()
    return {
        "method":    "dynamic",
        "q_id":      q_id,
        "label":     label,
        "choice":    st["final"]["final_choice"],
        "reasoning": st["final"]["final_reasoning"],
        "raw_state": st,
    }


async def run_special_generic(q_text: str, q_id: str, label: str) -> dict:
    mgr = Manager(q_text, q_id, label, n_specialists=3, n_generic=2, static_specialists=[])
    st  = await mgr.run_hybrid()
    return {
        "method":    "hybrid_special_generic",
        "q_id":      q_id,
        "label":     label,
        "choice":    st["final"]["final_choice"],
        "reasoning": st["final"]["final_reasoning"],
        "raw_state": st,
    }


async def run_static_dynamic(q_text: str, q_id: str, label: str) -> dict:
    mgr = Manager(q_text, q_id, label, n_specialists=5, n_generic=0, static_specialists=[])
    st  = await mgr.run_hybrid()
    return {
        "method":    "static_dynamic",
        "q_id":      q_id,
        "label":     label,
        "choice":    st["final"]["final_choice"],
        "reasoning": st["final"]["final_reasoning"],
        "raw_state": st,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────
async def process_row(row: pd.Series) -> List[dict]:
    q_text = f"{row['question']}\n\n{str(row['choice'])}"
    q_id  = str(row["qn_num"])
    label = str(row["ground_truth"])

    runners = [
        run_baseline,
        run_generic,
        run_dynamic,
        run_special_generic,
        # run_static_dynamic,
    ]
    # names   = ["baseline_zs", "generic", "dynamic", "hybrid_special_generic", "static_dynamic"]
    names   = ["baseline_zs", "generic", "dynamic", "hybrid_special_generic"]

    tasks   = [fn(q_text, q_id, label) for fn in runners]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out: List[dict] = []
    for name, res in zip(names, results):
        if isinstance(res, Exception):
            logger.error("❌ %s failed for QID %s – %s", name, q_id, res)
            out.append({"method": name, "q_id": q_id, "label": label, "error": str(res)})
        else:
            out.append(res)
    return out


async def process_dataset(df: pd.DataFrame) -> List[dict]:
    logger.info("Processing %d questions …", len(df))
    all_results: List[dict] = []
    for _, row in df.iterrows():
        all_results.extend(await process_row(row))
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
#  Entry‑point
# ──────────────────────────────────────────────────────────────────────────────
async def main() -> None:
    # 1) load dataset  (adjust the path as necessary)
    df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/Input/filtered_merged_QA.csv", lineterminator="\n")

    # 2) run the multi‑agent protocols
    results = await process_dataset(df)

    # 3) save
    out_path = "/home/yl3427/cylab/SOAP_MA/Output/MedicalQA/medical_QA_MA_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("✅ Saved → %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s — %(levelname)s — %(message)s",
        datefmt = "%Y‑%m‑%d %H:%M:%S",
        handlers=[
            logging.FileHandler("log/0530_medical_QA_MA.log", "w"),
            logging.StreamHandler(),
        ],
    )
    asyncio.run(main())
