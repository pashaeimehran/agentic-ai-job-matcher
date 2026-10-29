"""Deterministic, rule-based stand-in for a tool-calling model (fixture / zero-key demo mode).

It is NOT an LLM. It maps a few request shapes to tool calls with fixed rules
and writes answers by formatting the tool observations, so the real LangGraph
agent, ToolRegistry, retrieval and deterministic matching run end to end
without a paid provider. It does not simulate language understanding, model
non-determinism or open-ended answers.

Handled request shapes:
  - a known job_id + "save"            -> save_job
  - a known job_id + details wording   -> get_job_details
  - one or more job_ids                -> calculate_match
  - fit / match / best / suit wording  -> search_jobs, then calculate_match on the top results
  - find / search / jobs wording       -> search_jobs
  - CV / experience / skills wording   -> search_cv
  - anything else                      -> a short capabilities answer without tools
"""
from __future__ import annotations

import json
import re
from typing import Sequence

from src.llm.provider import LLMUsage
from src.llm.tool_calling import AssistantTurn, ChatMessage, ConversationItem, ToolCall, ToolResultMessage, ToolSpec

_JOB_ID = re.compile(r"\b((?:fixture|match)-[0-9]{2}(?:-[a-z0-9]+)*)\b", re.I)
_LOCATION = re.compile(r"\bin\s+([A-ZÄÖÜ][\w\-äöüß]+(?:\s+[A-ZÄÖÜ][\w\-äöüß]+)?)")
_ROLE = re.compile(r"(?:find|search(?: for)?|show(?: me)?|any|are there any|list)?\s*(?:me\s+)?"
                   r"(?P<role>[\w+#./ -]{2,80}?)\s+(?:jobs?|positions?|roles?|openings?)\b", re.I)
_MATCH_WORDS = ("fit", "match", "suit", "best", "compatib", "qualif", "score", "meet", "lack", "missing", "gap",
                "weakness", "apply")
_SEARCH_WORDS = ("find", "search", "jobs", "job ", "positions", "openings", "roles", "vacanc")
_DETAIL_WORDS = ("detail", "language", "posting", "describe")
_CV_WORDS = ("my cv", "my resume", "my experience", "my skills", "do i have", "my background", "my german",
             "my english", "my language")
_QUESTION_WORDS = {"what", "which", "is", "are", "my", "in", "the", "a", "an", "do", "does", "i", "have",
                   "cv", "resume", "of", "me", "tell", "about", "show", "please", "how", "much", "say", "says"}
FIXTURE_NOTE = "(Fixture mode: deterministic scripted responses, no LLM.)"


class FixtureToolCallingProvider:
    provider_name = "fixture"
    model = "fixture-scripted"
    model_parameters: dict = {}

    def complete_with_usage(self, messages: Sequence[ConversationItem],
                            tools: Sequence[ToolSpec]) -> tuple[AssistantTurn, LLMUsage]:
        """No tokens are consumed; zero usage lets cost accounting report 0.0 instead of unknown."""
        return self.complete(messages, tools), LLMUsage()

    def complete(self, messages: Sequence[ConversationItem], tools: Sequence[ToolSpec]) -> AssistantTurn:
        available = {t.name for t in tools}
        last_user = max(i for i, m in enumerate(messages) if isinstance(m, ChatMessage) and m.role == "user")
        request = messages[last_user].content
        observations = [m for m in messages[last_user + 1:] if isinstance(m, ToolResultMessage)]
        turn = sum(isinstance(m, AssistantTurn) for m in messages[last_user + 1:])
        if not observations:
            call = self._first_call(request)
            if call is None or call[0] not in available:
                return AssistantTurn(text=f"{FIXTURE_NOTE} I can search the job catalog, show job details, "
                                          "search your CV, calculate deterministic CV-job match scores and "
                                          "save jobs. Try: 'Find AI Engineer jobs in Germany that fit my CV'.")
            return self._call(turn, *call)
        last = json.loads(observations[-1].content)
        if (last.get("tool_name") == "search_jobs" and last.get("status") == "ok" and _wants_match(request)
                and "calculate_match" in available):
            job_ids = [job["job_id"] for job in last["data"]["jobs"][:3]]
            return self._call(turn, "calculate_match", {"job_ids": job_ids})
        return AssistantTurn(text=f"{FIXTURE_NOTE}\n{_render(json.loads(observations[-1].content))}")

    @staticmethod
    def _call(turn: int, name: str, arguments: dict) -> AssistantTurn:
        return AssistantTurn(tool_calls=[ToolCall(call_id=f"fx-{turn}-{name}", name=name,
                                                  arguments=json.dumps(arguments))])

    @staticmethod
    def _first_call(request: str) -> tuple[str, dict] | None:
        lowered = request.casefold()
        job_ids = list(dict.fromkeys(m.lower() for m in _JOB_ID.findall(request)))
        if job_ids:
            if "save" in lowered:
                return "save_job", {"job_id": job_ids[0]}
            if any(w in lowered for w in _DETAIL_WORDS) and not _wants_match(request):
                return "get_job_details", {"job_id": job_ids[0]}
            return "calculate_match", {"job_ids": job_ids[:10]}
        searching = any(w in lowered for w in _SEARCH_WORDS)
        if any(w in lowered for w in _CV_WORDS) and not _wants_match(request) and not searching:
            return "search_cv", {"query": _cv_query(request)}
        if _wants_match(request) or any(w in lowered for w in _SEARCH_WORDS):
            args: dict = {"query": _role(request)}
            location = _LOCATION.search(request)
            if location:
                args["location"] = location.group(1)
            return "search_jobs", args
        return None


def _wants_match(request: str) -> bool:
    # Job IDs such as "match-03-robotics" must not count as match wording.
    text = _JOB_ID.sub(" ", request).casefold()
    return any(w in text for w in _MATCH_WORDS)


def _cv_query(request: str) -> str:
    words = [w for w in re.findall(r"[\w+#.-]+", request) if w.casefold() not in _QUESTION_WORDS]
    return " ".join(words)[:500] or request[:500]


def _role(request: str) -> str:
    match = _ROLE.search(request)
    role = match.group("role").strip() if match else request
    return role[:200] or request[:200]


def _render(observation: dict) -> str:
    tool, status, data = observation.get("tool_name"), observation.get("status"), observation.get("data") or {}
    if status == "error":
        return f"The {tool} tool reported an error: {observation['error']['message']}"
    if status == "empty":
        return data.get("message") or "Nothing was found."
    if tool == "calculate_match":
        lines = ["Deterministic match results (scores are calculated by Python code, not estimated):"]
        for match in data["matches"]:
            lines.append(f"{match['rank']}. {match['role']} ({match['job_id']}): {match['score']['value']}/100")
            lines.append(f"   Supported: {', '.join(match['supported']) or 'none'}")
            lines.append(f"   Unmet: {', '.join(match['unmet']) or 'none'}")
            lines.append(f"   Unknown (no evidence in the CV, not a confirmed gap): "
                         f"{', '.join(match['unknown']) or 'none'}")
        return "\n".join(lines)
    if tool == "search_jobs":
        lines = [f"Found {data['total_matches']} matching job(s):"]
        lines += [f"- {j['job_id']}: {j['role']} ({j['location'] or 'location not stated'}); "
                  f"skills: {', '.join(j['skills'])}" for j in data["jobs"]]
        return "\n".join(lines)
    if tool == "get_job_details":
        posting = data["posting"]
        levels = data.get("language_levels") or {}
        years = posting.get("experience_years")
        experience = f"{years} years" if years else "not stated"
        languages = ", ".join(f"{name} {levels.get(name, '')}".strip() for name in posting["language"]) or "none"
        return (f"{posting['role']} ({data['job_id']}), {posting.get('location') or 'location not stated'}. "
                f"Skills: {', '.join(posting['skills']) or 'none'}. Languages: {languages}. "
                f"Experience: {experience}.")
    if tool == "search_cv":
        return "\n".join(["CV excerpts:"] + [f"- [{e['section']}] {e['text']}" for e in data["evidence"]])
    if tool == "save_job":
        return f"Job {data['job_id']} is {'saved' if data['status'] == 'saved' else 'already saved'}."
    return json.dumps(data)[:500]
