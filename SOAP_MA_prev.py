#########################################################################
# multi_agent_diagnosis.py 
###############################################################################
"""LLM-driven multi-agent system for analysing SOAP progress notes and
predicting whether a patient has a given problem.

Supports three evaluation modes
--------------------------------
1. Baseline            - single zero-shot agent
2. Generic-only        - N generic agents
3. Dynamic-specialist  - LLM-proposed panel of M specialists
4. **Hybrid**          - user-fixed static specialists ➊ + additional
                         LLM-proposed specialists ➋ + K generic agents
"""

from __future__ import annotations
###############################################################################
#                                   Imports                                   #
###############################################################################
import asyncio, inspect, json, logging, math, os, re
from typing import (Any, Dict, List, Literal, Optional, Union, get_args,
                    get_origin)

import chromadb
import pandas as pd
from chromadb.config import Settings
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from rag_retrieve import create_documents, hybrid_query
from utils import count_llama_tokens, safe_json_load

###############################################################################
#                                Constants                                    #
###############################################################################
LLAMA3_70B_MAX_TOKENS = 24_000
logger = logging.getLogger(__name__)

selected_problems = [
    'congestive heart failure',
    'sepsis',
    'acute kidney injury',
]

# --- hybrid helper ---------------------------------------------------- #
STATIC_BY_PROBLEM = {
    "congestive heart failure": ["Cardiologist", "Cardiac electrophysiologist"],
    "sepsis": ["Infectious Disease Specialist", "Intensive Care Specialist"],
    "acute kidney injury": ["Nephrologist", "Intensive Care Specialist"],
    # ...
}

###############################################################################
#                           Pydantic response schema                          #
###############################################################################
class Response(BaseModel):
    reasoning: str = Field(...,
        description="Step-by-step reasoning leading to your choice.")
    choice: Literal["Yes", "No"] = Field(...,
        description="Your choice indicating whether the patient has the problem.")

###############################################################################
#                               Core LLMAgent                                 #
###############################################################################
class LLMAgent:
    """Base wrapper around an OpenAI-compatible chat model with automatic
    in-place summarisation to stay within the context window."""

    def __init__(self,
        system_prompt: str,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        client      = AsyncOpenAI(base_url="http://localhost:8000/v1",
                                  api_key="dummy"),
        max_tokens: int = LLAMA3_70B_MAX_TOKENS,
        summarisation_threshold: float = 0.80):

        self.model_name  = model_name
        self.client      = client
        self.messages    = [{"role": "system", "content": system_prompt}]
        self.max_tokens  = max_tokens
        self.token_thres = int(max_tokens * summarisation_threshold)
        logger.info("[%s] init – token limit %d, summarise ≥%d",
                    self.__class__.__name__, max_tokens, self.token_thres)

    # --------------------------------------------------------------------- #
    # Iterative summarisation                                   #
    # --------------------------------------------------------------------- #
    
    # ------------------------------------------------------------------ #
    # 1A.  _summarise_once — now accepts max_chars                       #
    # ------------------------------------------------------------------ #
    async def _summarise_once(self, text: str,
                            max_chars: Optional[int] = None) -> Optional[str]:
        """
        Return one-shot summary of *text*.
        If `max_chars` is given, the prompt tells the model to keep the summary
        ≤ that many characters.
        """
        length_rule = (f"Do **not** exceed {max_chars} characters."
                    if max_chars is not None
                    else "Do **not** exceed 1000 words.")

        prompt = (
            "Summarise the following message concisely, preserving every key "
            f"fact and reasoning step. {length_rule}\n\n"
            "<<<MESSAGE_START>>>\n"
            f"{text}\n<<<MESSAGE_END>>>"
        )

        try:
            resp = await self.client.chat.completions.create(
                model      = self.model_name,
                messages   = [
                    {"role": "system", "content": "You are a summarisation assistant."},
                    {"role": "user",   "content": prompt}],
                temperature = 0.1,
                max_tokens  = 1500)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Summarisation call failed: %s", e)
            return None


    # ------------------------------------------------------------------ #
    # 1B.  _summarize_history — thread max_chars through                 #
    # ------------------------------------------------------------------ #
    async def _summarize_history(self,
                                inplace: bool = True,
                                message: str = "",
                                max_chars: Optional[int] = None
                                ) -> Union[bool, str]:
        """
        Summarise either self.messages (in-place) or the external *message*.
        *max_chars* (if given) is forwarded to the one-shot summariser, letting
        the caller control the final length instead of hard-cutting later.
        """
        if inplace:
            if len(self.messages) < 3:
                logger.warning("Not enough messages to summarise.")
                return False

            tried: set[int] = set()
            while count_llama_tokens(self.messages) >= self.token_thres:
                idx_len = sorted(
                    ((i, count_llama_tokens([m]))
                    for i, m in enumerate(self.messages)
                    if m["role"] != "system" and i not in tried),
                    key=lambda t: t[1],
                    reverse=True)

                if not idx_len:
                    logger.error("No further messages left to summarise.")
                    return False

                idx      = idx_len[0][0]
                summary  = await self._summarise_once(self.messages[idx]["content"],
                                                    max_chars=max_chars)
                if summary is None:
                    tried.add(idx)
                    continue
                if count_llama_tokens([{"role": "assistant", "content": summary}]) \
                >= count_llama_tokens([self.messages[idx]]):
                    tried.add(idx)
                    continue

                self.messages[idx]["content"] = f"[Summary] {summary}"
                logger.info("Summarised message #%d → total %d tokens",
                            idx, count_llama_tokens(self.messages))
            return True

        # ---- external string path ----
        if not message:
            logger.warning("No message supplied for external summarisation.")
            return False

        summary = await self._summarise_once(message, max_chars=max_chars)
        if summary is None:
            return False

        while (count_llama_tokens(summary) >= self.token_thres) \
            or (max_chars and len(summary) > max_chars):
            summary = await self._summarise_once(summary, max_chars=max_chars)
            if summary is None:
                return False
        return summary

    # --------------------------------------------------------------------- #
    # LLM call helpers (unchanged except var rename)                        #
    # --------------------------------------------------------------------- #
    async def _llm_call_with_tools(self, params: dict,
                                   available_tools: dict) -> Any:
        if count_llama_tokens(params["messages"]) > self.token_thres:
            await self._summarize_history()
            params["messages"] = self.messages
            if count_llama_tokens(params["messages"]) > self.max_tokens:
                logger.error("Summarisation failed – still over limit.")
                return "Error: context limit exceeded."
        iter_ct = 0
        try:
            resp = await self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error("Initial LLM+tools call failed: %s", e)
            return None
        msg = resp.choices[0].message

        while not getattr(msg, "tool_calls", []):
            self.messages.append({"role": "assistant", "content": msg.content})
            self.messages.append({"role": "system",
                                  "content": "Reminder: you must call a tool."})
            iter_ct += 1
            if iter_ct > 3:
                return msg.content
            try:
                resp = await self.client.chat.completions.create(**params)
            except Exception as e:
                logger.error("Retry LLM+tools failed: %s", e)
                return None
            msg = resp.choices[0].message

        # execute tool calls …
        self.messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
        for call in msg.tool_calls:
            args = safe_json_load(call.function.arguments)
            if args is None or call.function.name not in available_tools:
                continue
            try:
                result = available_tools[call.function.name](**args)
            except Exception as e:
                logger.error("Tool '%s' failed: %s", call.function.name, e)
                return msg.content
            self.messages.append({"role": "tool", "name": call.function.name,
                                  "tool_call_id": call.id, "content": result})

        try:
            resp = await self.client.chat.completions.create(**params)
            return resp.choices[0].message.content
        except Exception as e:
            logger.error("Post-tool LLM call failed: %s", e)
            return None

    async def llm_call(self, user_prompt: str, temperature: float = .3,
                       guided_: dict = None,
                       tools_descript: List[dict] = None,
                       available_tools: dict = None) -> Any:
        if (count_llama_tokens(self.messages + [{"role":"user","content":user_prompt}])
                > self.token_thres):
            ok = await self._summarize_history()
            if not ok and count_llama_tokens(self.messages) > self.max_tokens:
                raise ValueError("Context limit even after summarisation.")
        self.messages.append({"role": "user", "content": user_prompt})
        params = {"model": self.model_name, "messages": self.messages,
                  "temperature": temperature}
        if guided_:
            params["extra_body"] = guided_
        if tools_descript:
            params["tools"] = tools_descript
            return await self._llm_call_with_tools(params, available_tools)
    
        resp = await self.client.chat.completions.create(**params)
        return resp.choices[0].message.content

    def append_message(self, content: Any, role="assistant"):
        self.messages.append({"role": role, "content": str(content)})

###############################################################################
#                         Concrete agent subclasses                           #
###############################################################################
class BaselineZS(LLMAgent):
    def __init__(self):
        super().__init__("You are a clinical reasoning assistant.")
        self.schema = Response.model_json_schema()

    async def analyze_note(self, note: str, problem: str):
        user_prompt = (
            "Read the patient note and decide whether the patient has the "
            "specified problem.\n"
            f"<<<PATIENT NOTE>>>\n{note}\n<<<END NOTE>>>\n\n"
            f"Question: Does this patient have {problem}?\n"
            "Please give your reasoning, then the choice ('Yes' or 'No').")
        raw = await self.llm_call(
            user_prompt, temperature=0.1,
            guided_={"guided_json": self.schema})
        return safe_json_load(raw)

# ------------------------------------------------------------------------- #
# GenericAgent (unchanged except peer merge)                               #
# ------------------------------------------------------------------------- #
class GenericAgent(LLMAgent):
    def __init__(self, agent_id: str, state: dict):
        self.agent_id = str(agent_id)
        self.state    = state
        self.state.setdefault("generic_agents", {})[self.agent_id] = {}
        self.round    = 0
        self.schema   = Response.model_json_schema()
        super().__init__(
            "You are a collaborating diagnostic agent in a multi-agent AI system.")

    async def analyze_note(self, note: str, problem: str):
        self.round += 1
        prompt = (
            "Read the patient note and decide whether the patient has the "
            "specified problem.\n"
            f"<<<PATIENT NOTE>>>\n{note}\n<<<END NOTE>>>\n\n"
            f"Question: Does this patient have {problem}?\n"
            "Please give your reasoning, then the choice ('Yes' or 'No').")
        raw = await self.llm_call(
            prompt, temperature=0.1,
            guided_={"guided_json": self.schema})
        parsed = safe_json_load(raw)
        self.append_message(raw)
        self.state["generic_agents"][self.agent_id][f"round_{self.round}"] = parsed
        return parsed

    async def debate(self):
        self.round += 1
        # compile *all* peers (generic + specialists)
        peers: Dict[str, Any] = {}
        for a, hist in self.state["generic_agents"].items():
            if a != self.agent_id and f"round_{self.round-1}" in hist:
                peers[a] = hist[f"round_{self.round-1}"]
        # specialists
        for p in [k for k in self.state if k.startswith("panel_")]:
            for role, info in self.state[p]["Collected Specialists"].items():
                if f"round_{self.round-1}" in info["answer_history"]:
                    peers[role] = info["answer_history"][f"round_{self.round-1}"]

        prompt = (
            "Here are your peers’ previous answers:\n"
            f"{json.dumps(peers, indent=2)}\n\n"
            "Review their reasoning. Considering their input and your own "
            "analysis, decide whether to revise your answer.\n\n"
            "Return your updated reasoning and final choice ('Yes' or 'No').")
        raw = await self.llm_call(
            prompt, temperature=0.1,
            guided_={"guided_json": self.schema})
        parsed = safe_json_load(raw)
        self.append_message(raw)
        self.state["generic_agents"][self.agent_id][f"round_{self.round}"] = parsed
        return parsed

# ------------------------------------------------------------------------- #
# DynamicSpecialist – debate now includes generic peers                    #
# ------------------------------------------------------------------------- #
class DynamicSpecialist(LLMAgent):
    def __init__(self, specialist: str, panel_id: int, state: dict):
        self.specialist = specialist
        self.panel_id   = panel_id
        self.state      = state
        record          = state[f"panel_{panel_id}"]["Collected Specialists"][specialist]
        self.expertise  = record["expertise"]
        self.answer_hist= record["answer_history"]
        self.round      = 0
        self.schema     = Response.model_json_schema()
        super().__init__(
            f"You are a {specialist}. Your expertise:\n{self.expertise}\n"
            "Analyze the patient's condition from your specialty's viewpoint.")

    async def analyze_note(self, note: str, problem: str):
        self.round += 1
        prompt = (
            "Read the patient note and decide whether the patient has the "
            "specified problem.\n"
            f"<<<PATIENT NOTE>>>\n{note}\n<<<END NOTE>>>\n\n"
            f"Question: Does this patient have {problem}?\n"
            "Please give your reasoning, then the choice ('Yes' or 'No').")
        raw = await self.llm_call(
            prompt, temperature=0.1,
            guided_={"guided_json": self.schema})
        parsed = safe_json_load(raw)
        self.append_message(raw)
        self.answer_hist[f"round_{self.round}"] = parsed
        return parsed

    async def debate(self):
        self.round += 1
        peers: Dict[str, Any] = {}
        # other specialists
        for role, info in self.state[f"panel_{self.panel_id}"]["Collected Specialists"].items():
            if role != self.specialist and f"round_{self.round-1}" in info["answer_history"]:
                peers[role] = info["answer_history"][f"round_{self.round-1}"]
        # generic agents
        for a, hist in self.state.get("generic_agents", {}).items():
            if f"round_{self.round-1}" in hist:
                peers[a] = hist[f"round_{self.round-1}"]

        prompt = (
            "Here are your peers’ previous answers:\n"
            f"{json.dumps(peers, indent=2)}\n\n"
            "Review their reasoning. Considering their input and your own "
            "analysis, decide whether to revise your answer.\n\n"
            "Return your updated reasoning and final choice ('Yes' or 'No').")
        raw = await self.llm_call(
            prompt, temperature=0.1,
            guided_={"guided_json": self.schema})
        parsed = safe_json_load(raw)
        self.append_message(raw)
        self.answer_hist[f"round_{self.round}"] = parsed
        return parsed

###############################################################################
#                             Manager                                         #
###############################################################################
class Manager(LLMAgent):
    def __init__(
        self,
        note: str,
        hadm_id: str,
        problem: str,
        label: str,
        n_specialists: Union[int, Literal["auto"]] = "auto",
        n_generic_agents: int = 0,
        static_specialists: Optional[List[str]] = None,
        consensus_threshold: float = 0.8,
        max_consensus_attempts: int = 3,
        max_assignment_attempts: int = 2,
        single_step: bool = False,
        each_agent_summary_char_limit: int = 100,
    ):
        super().__init__(
            "You are the manager of a multi-agent diagnostic system. "
            "Coordinate sub-agents to reach a final decision.")

        # ---- store arguments -------------------------------------------------
        self.note              = note
        self.hadm_id           = hadm_id
        self.problem           = problem
        self.label             = label
        self.n_specialists     = n_specialists
        self.n_generic_agents  = n_generic_agents
        self.static_specs      = static_specialists or []
        self.cons_thresh       = consensus_threshold
        self.max_consensus     = max_consensus_attempts
        self.max_assign        = max_assignment_attempts
        self.single_step       = single_step     # ← NEW
        self.each_agent_summary_char_limit = each_agent_summary_char_limit

        self.assign_attempts   = 0

        # ---- shared scratch-pad ---------------------------------------------
        self.state: Dict[str, Any] = {
            "note": note, "hadm_id": hadm_id, "problem": problem,
            "label": label, "generic_agents": {}, "final": {}
        }


    # -------------------- specialist panel assignment -------------------- #
    async def _assign_specialists(self) -> None:
        """Create one specialist panel and record it in self.state.

        • If self.single_step == False  →  two-step workflow
        • If self.single_step == True   →  single merged prompt
        """
        self.assign_attempts += 1
        panel_id = self.assign_attempts
        self.state[f"panel_{panel_id}"] = {
            "Initially Identified Specialties": [],
            "Collected Specialists": {}
        }

        # ────────────────────────── STEP 1 ──────────────────────────────────────
        # Decide *which* specialties we need (skipped in single-step mode only
        # when n_specialists is an explicit integer and no dynamics are required).
        specialties: List[str] = list(self.static_specs)

        additional_needed: Union[int, Literal["auto"]]
        if self.n_specialists == "auto":
            additional_needed = "auto"
        else:
            additional_needed = max(self.n_specialists - len(self.static_specs), 0)

        if not self.single_step and additional_needed != 0:
            # Ask LLM for the extra specialties
            ask_specialties = (
                "Below are the **Subjective (S)** and **Objective (O)** sections of a "
                "patient’s SOAP note:\n\n"
                f"<S+O NOTE>\n{self.note}\n</S+O NOTE>\n\n"
                f"Our task is to determine **whether the patient has «{self.problem}»**.\n"
                f"Specialties already included: {self.static_specs or '(none)'}.\n"
                "Please list any **additional medical specialties** whose expertise would "
                f"help decide whether the patient has «{self.problem}»."
            )

            if additional_needed == "auto":
                class SpecialtyList(BaseModel):
                    specialties: List[str]
                reply = await self.llm_call(
                    ask_specialties,
                    guided_={"guided_json": SpecialtyList.model_json_schema()})
                specialties.extend(safe_json_load(reply)["specialties"])

            else:  # fixed number of extra specialties
                fld = {f"specialty_{i+1}": (str, ...)
                    for i in range(additional_needed)}
                Requested = create_model("RequestedSpecialties", **fld)
                reply = await self.llm_call(
                    ask_specialties,
                    guided_={"guided_json": Requested.model_json_schema()})
                specialties.extend([safe_json_load(reply)[f"specialty_{i+1}"]
                                    for i in range(additional_needed)])

        self.state[f"panel_{panel_id}"]["Initially Identified Specialties"] = specialties

        # ────────────────────────── STEP 2 ──────────────────────────────────────
        # Convert those specialties into concrete specialist personas.
        if self.single_step:
            # One prompt: returns the full panel directly.
            total_needed = (len(specialties) if self.n_specialists != "auto"
                            else f"at least {len(specialties)}")

            ask_panel = (
                "Below are only the S and O sections of a patient's SOAP note:\n\n"
                f"<S+O NOTE>\n{self.note}\n</S+O NOTE>\n\n"
                f"We must decide **whether the patient has «{self.problem}»**.\n"
                f"Specialties that must appear (already fixed): "
                f"{self.static_specs or '(none)'}.\n"
                f"Create a panel of {total_needed} specialists in total.  For *each* "
                "specialist return an object with:\n"
                "  • `specialist`  – the full job title you want the agent to role-play\n"
                 "  • `expertise`  – 1‒3 short phrases explaining how that role is "
                "                    helpful for our yes/no decision."
            )

            class SpecialistDescription(BaseModel):
                specialist: str
                expertise:  List[str]

            fld = {f"specialist_{i+1}": (SpecialistDescription, ...)
                for i in range(len(specialties))}
            PanelOut = create_model("SpecialistPanel", **fld)

            reply       = await self.llm_call(
                ask_panel,
                guided_={"guided_json": PanelOut.model_json_schema()})
            panel_json  = safe_json_load(reply)
            panel_dict  = {v["specialist"]: v["expertise"]
                        for v in panel_json.values()}

        else:
            # Two-step: we already have `specialties`, now turn each into a persona.
            ask_panel = (
                "We will run a multi-agent debate to answer one question:\n"
                f"» **Does the patient described in the S+O note have «{self.problem}»?**\n\n"
                "The required specialties are:\n"
                f"{specialties}\n\n"
                "For each specialty provide an object with:\n"
                "  • `specialist` – the full job title for the agent to play\n"
                "  • `expertise`  – 1‒3 short phrases explaining how that role is "
                "                    helpful for our yes/no decision."
            )

            class SpecialistDescription(BaseModel):
                specialist: str
                expertise:  List[str]

            fld = {f"specialist_{i+1}": (SpecialistDescription, ...)
                for i in range(len(specialties))}
            PanelOut = create_model("SpecialistPanel", **fld)

            reply      = await self.llm_call(
                ask_panel,
                guided_={"guided_json": PanelOut.model_json_schema()})
            panel_json = safe_json_load(reply)
            panel_dict = {v["specialist"]: v["expertise"]
                        for v in panel_json.values()}

        # ────────────────────────── COMMIT TO STATE ─────────────────────────────
        for role, expertise in panel_dict.items():
            self.state[f"panel_{panel_id}"]["Collected Specialists"][role] = {
                "expertise": expertise,
                "answer_history": {}
            }

    async def _panel_summary(self, panel_id: int) -> str:
        """
        Summarise each participant’s latest reasoning.  The model itself is asked
        to keep the summary ≤ self.summary_char_limit characters (if set), so no
        post-hoc slicing is needed.
        """
        # -------- identify latest round id ----------
        round_ids = []
        for info in self.state[f"panel_{panel_id}"]["Collected Specialists"].values():
            round_ids.extend(int(k.split('_')[1]) for k in info["answer_history"])
        for hist in self.state.get("generic_agents", {}).values():
            round_ids.extend(int(k.split('_')[1]) for k in hist)
        if not round_ids:
            return "(no valid answers to summarise)"
        last_round = max(round_ids)

        # -------- collect (name, choice, reasoning) tuples ----------
        blobs: list[tuple[str, str, str]] = []
        # specialists
        for role, info in self.state[f"panel_{panel_id}"]["Collected Specialists"].items():
            tag = f"round_{last_round}"
            if tag in info["answer_history"]:
                ans = info["answer_history"][tag]
                blobs.append((role, ans["choice"], ans["reasoning"]))
        # generic agents
        for ag, hist in self.state.get("generic_agents", {}).items():
            tag = f"round_{last_round}"
            if tag in hist:
                ans = hist[tag]
                blobs.append((ag, ans["choice"], ans["reasoning"]))

        # -------- summarise concurrently ----------
        async def compress(txt: str) -> str:
            out = await self._summarize_history(
                inplace=False,
                message=txt,
                max_chars=self.each_agent_summary_char_limit      # ← pass limit
            )
            return out if isinstance(out, str) else txt.strip()

        summaries = await asyncio.gather(*(compress(r) for _, _, r in blobs))

        # -------- compose bullet list ----------
        lines = [
            f"- **{name}** → {choice} | {summ}"
            for (name, choice, _), summ in zip(blobs, summaries)
        ]
        return "\n".join(lines)



    # ---------- consensus helpers (works for hybrid or specialist-only) ---------- #
    def _check_consensus(self, panel_id: int, round_id: int,
                         total_agents: int) -> Optional[str]:
        count: Dict[str,int] = {}
        # specialists
        for role, info in self.state[f"panel_{panel_id}"]["Collected Specialists"].items():
            if f"round_{round_id}" in info["answer_history"]:
                ch = info["answer_history"][f"round_{round_id}"]["choice"]
                count[ch] = count.get(ch,0)+1
        # generic agents
        for ag, hist in self.state.get("generic_agents", {}).items():
            if f"round_{round_id}" in hist:
                ch = hist[f"round_{round_id}"]["choice"]
                count[ch] = count.get(ch,0)+1
        for ch, c in count.items():
            if c >= math.ceil(total_agents * self.cons_thresh):
                return ch
        return None

    # --------------------------------------------------------------------- #
    # HYBRID run                                                            #
    # --------------------------------------------------------------------- #
    async def run_hybrid(self):
        """Static + dynamic specialists + generic agents (full hybrid)."""
        while self.assign_attempts < self.max_assign:
            # ——— If this is NOT the first attempt, explain why we are retrying ———
            if self.assign_attempts:
                prev   = self.assign_attempts                      # panel just finished
                summ   = await self._panel_summary(prev)
                self.append_message(
                    "⚠️ The previous specialist panel failed to reach consensus.\n\n"
                    "**Their final positions were:**\n"
                    f"{summ}\n\n"
                    "Please assemble a *new* mix of specialties to resolve the disagreement.",
                    role="system")

            # ——— Build a fresh panel ———
            await self._assign_specialists()
            panel_id = self.assign_attempts

            specialists = [DynamicSpecialist(role, panel_id, self.state)
                        for role in self.state[f"panel_{panel_id}"]["Collected Specialists"]]
            generics    = [GenericAgent(f"generic_{i+1}", self.state)
                        for i in range(self.n_generic_agents)]

            agents      = specialists + generics
            # total_n     = len(agents)

            # ——— Round-1 analysis ———
            tasks = [a.analyze_note(self.note, self.problem) for a in agents]
            # await asyncio.gather(*tasks)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for ag, res in zip(agents, results):
                if isinstance(res, Exception):
                    logger.exception("Agent %s crashed during analyse_note", ag)
            choice = self._check_consensus(panel_id, 1, len(agents))
            if choice:
                self.state["final"] = {"final_choice": choice,
                                    "final_reasoning": "Consensus reached"}
                return self.state

            # ——— Debate rounds ———
            for r in range(2, self.max_consensus + 1):
                # await asyncio.gather(*[a.debate() for a in agents])
                results = await asyncio.gather(
                    *[a.debate() for a in agents], return_exceptions=True)
                for ag, res in zip(agents, results):
                    if isinstance(res, Exception):
                        logger.exception("Agent %s crashed during debate", ag)
                choice = self._check_consensus(panel_id, r, len(agents))
                if choice:
                    self.state["final"] = {"final_choice": choice,
                                        "final_reasoning": "Consensus reached"}
                    return self.state

            # … otherwise loop again (next panel) until max_assign is hit.

        # ——— Fallback: aggregator ———
        hist = {k: v for k, v in self.state.items() if k not in ("label", "final")}
        await self._aggregate(hist)
        return self.state


    # ---------------- Aggregator ----------------------------- #
    async def _aggregate(self, chat_history):
        prompt = (
            "The sub-agents failed to reach consensus. Below is the entire "
            "conversation history:\n\n"
            f"{json.dumps(chat_history, indent=4)}\n\n"
            "Analyze their reasoning and provide:\n"
            "1) A concise explanation of how you reached the final decision\n"
            "2) The single best-supported choice: 'Yes' or 'No'")
        class Out(BaseModel):
            final_reasoning: str
            final_choice:   Literal["Yes","No"]
        raw = await self.llm_call(prompt, temperature=0.1,
                                  guided_={"guided_json": Out.model_json_schema()})
        parsed = safe_json_load(raw)
        self.state["final"] = parsed
        return parsed

###############################################################################
#                    Convenience wrappers for each mode                       #
###############################################################################
async def run_baseline(note, hadm_id, problem, label):
    zs = BaselineZS()
    out = await zs.analyze_note(note, problem)
    return {"method":"baseline_zs","hadm_id":hadm_id,"label":label,
            "choice":out["choice"] if out else "ERROR",
            "reasoning":out["reasoning"] if out else "failed","raw_state":out}

async def run_generic(note, hadm_id, problem, label):
    mgr = Manager(note, hadm_id, problem, label,
                  n_specialists=0, n_generic_agents=5)
    st  = await mgr.run_hybrid()   # hybrid but w/ 0 specialists
    return {"method":"generic","hadm_id":hadm_id,"label":label,
            "choice":st["final"]["final_choice"],"reasoning":st["final"]["final_reasoning"],
            "raw_state":st}

async def run_dynamic(note, hadm_id, problem, label):
    mgr = Manager(note, hadm_id, problem, label,
                  n_specialists=5, n_generic_agents=0)
    st  = await mgr.run_hybrid()   # hybrid but w/ 0 generics
    return {"method":"dynamic","hadm_id":hadm_id,"label":label,
            "choice":st["final"]["final_choice"],"reasoning":st["final"]["final_reasoning"],
            "raw_state":st}

async def run_special_generic(note, hadm_id, problem, label):
    mgr = Manager(note, hadm_id, problem, label,
        n_specialists=3, # 1 dynamic + 2 static
        n_generic_agents=2,
        static_specialists=STATIC_BY_PROBLEM.get(problem.lower(), [])
        )
    st = await mgr.run_hybrid()
    return {
        "method":    "hybrid_special_generic",
        "hadm_id":   hadm_id,
        "label":     label,
        "choice":    st["final"]["final_choice"],
        "reasoning": st["final"]["final_reasoning"],
        "raw_state": st
    }

async def run_static_dynamic(note, hadm_id, problem, label):
    static_specs = STATIC_BY_PROBLEM.get(problem.lower(), [])

    mgr = Manager(note, hadm_id, problem, label,
        n_specialists=5, # 3 dynamic + 2 static
        n_generic_agents=0,               # no generics
        static_specialists=static_specs
        )
    st = await mgr.run_hybrid()
    return {
        "method":       "static_dynamic",
        "hadm_id":      hadm_id,
        "label":        label,
        "choice":       st["final"]["final_choice"],
        "reasoning":    st["final"]["final_reasoning"],
        "raw_state":    st
    }
async def process_row(row, problem):
    note  = f"{row['Subjective']}\n{row['Objective']}"
    hadm  = row["File ID"]
    label = row["combined_summary"]

    try:
        return await asyncio.gather(
            run_baseline(     note, hadm, problem, label),
            run_generic(      note, hadm, problem, label),
            run_dynamic(      note, hadm, problem, label),
            run_special_generic(note, hadm, problem, label),
            run_static_dynamic(note, hadm, problem, label),
        )
    except Exception as e:
        logger.exception("⚠️  Skipping HADM %s – %s", hadm, e)
        return [{
                    "method": "SKIPPED",
                    "hadm_id": hadm,
                    "label": label,
                    "reason": str(e)
                }]

async def process_problem(df, problem):
    logger.info("Processing %s (%d rows)…", problem, len(df))
    results = []
    for _, row in df.iterrows():
        results.extend(await process_row(row, problem))
    out = f"/home/yl3427/cylab/SOAP_MA/Output/SOAP/hybrid/" \
          f"results_{problem.replace(' ','_')}.json"
    with open(out,"w") as f: json.dump(results,f,indent=2)
    logger.info("Saved → %s", out)

###############################################################################
#                                      main                                   #
###############################################################################
async def main():
    df = pd.read_csv("/home/yl3427/cylab/SOAP_MA/Input/SOAP_all_problems.csv",
                     lineterminator="\n")
    tasks = [asyncio.create_task(process_problem(df, p))
             for p in selected_problems]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("log/0425_MA_hybrid.log","w"),
            logging.StreamHandler()
        ]
    )
    asyncio.run(main())
