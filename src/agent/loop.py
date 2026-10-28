"""Framework-independent manual agent loop.

request -> model turn -> validated tool calls -> observations -> model turn ...
until the model answers without tool calls or a configured limit is reached.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.schemas import AgentConfig, AgentRunResult, ToolCallRecord
from src.llm.tool_calling import (ChatMessage, ConversationItem, ToolCallingLLMProvider,
                                  ToolResultMessage)
from src.tools.registry import ToolRegistry

logger = logging.getLogger("src.agent")

_SCORE_PATTERNS = (
    re.compile(r"(?<![\w.-])(\d{1,3}(?:\.\d+)?)\s*/\s*100\b"),
    # Lookarounds skip digits inside identifiers such as "match-01".
    re.compile(r"\bscor(?:e|ed|es)\b[^\d\n]{0,20}?(?<![\w.-])(\d{1,3}(?:\.\d+)?)(?![\w-])", re.I),
)


def _log(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


def ungrounded_scores(answer: str, records: list[ToolCallRecord]) -> list[str]:
    """Score-like numbers in the answer that no successful calculate_match returned."""
    allowed = {round(match["score"]["value"], 2)
               for r in records if r.tool_name == "calculate_match" and r.outcome.data
               for match in r.outcome.data["matches"]}
    found = [m for pattern in _SCORE_PATTERNS for m in pattern.findall(answer)]
    return sorted({value for value in found if round(float(value), 2) not in allowed})


class ManualAgent:
    def __init__(self, provider: ToolCallingLLMProvider, registry: ToolRegistry, context: Any,
                 config: AgentConfig = AgentConfig(), system_prompt: str = SYSTEM_PROMPT) -> None:
        self.provider, self.registry, self.context = provider, registry, context
        self.config, self.system_prompt = config, system_prompt

    def run(self, request: str, run_id: str | None = None) -> AgentRunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        cfg = self.config
        messages: list[ConversationItem] = [ChatMessage(role="system", content=self.system_prompt),
                                            ChatMessage(role="user", content=request)]
        specs = self.registry.specs()
        records: list[ToolCallRecord] = []
        # The request text is not logged: it may contain personal data.
        _log("agent_run_started", run_id=run_id, max_model_turns=cfg.max_model_turns,
             max_tool_calls=cfg.max_tool_calls, tools=[s.name for s in specs])

        def finish(status, turns, final=None, error=None) -> AgentRunResult:
            result = AgentRunResult(run_id=run_id, status=status, final_answer=final, model_turns=turns,
                                    tool_calls=records, error=error,
                                    ungrounded_scores=ungrounded_scores(final, records) if final else [])
            _log("agent_run_finished", run_id=run_id, status=status, model_turns=turns,
                 tool_calls=len(records), ungrounded_scores=len(result.ungrounded_scores))
            return result

        for step in range(1, cfg.max_model_turns + 1):
            try:
                reply = self.provider.complete(messages, specs)
            except Exception as exc:  # noqa: BLE001 - provider payloads are not echoed
                _log("model_turn_failed", run_id=run_id, step=step, exception=type(exc).__name__)
                return finish("provider_error", step, error=f"Model call failed ({type(exc).__name__}).")
            messages.append(reply)
            _log("model_turn", run_id=run_id, step=step, tool_calls=[c.name for c in reply.tool_calls],
                 has_text=bool(reply.text))
            if not reply.tool_calls:
                return finish("completed", step, final=reply.text or "")

            for call in reply.tool_calls:
                if len(records) >= cfg.max_tool_calls:
                    return finish("max_tool_calls_exceeded", step,
                                  error=f"Tool-call limit of {cfg.max_tool_calls} reached.")
                outcome = self.registry.execute(call.name, call.arguments, self.context, cfg.tool_retry,
                                                run_id=run_id, step=step)
                records.append(ToolCallRecord(
                    step=step, call_id=call.call_id, tool_name=call.name,
                    arguments_valid=outcome.error is None or outcome.error.code not in
                    ("invalid_arguments", "unknown_tool"),
                    status=outcome.status, error_code=outcome.error.code if outcome.error else None,
                    attempts=outcome.attempts, duration_ms=outcome.duration_ms, outcome=outcome))
                messages.append(ToolResultMessage(
                    call_id=call.call_id, name=call.name,
                    content=outcome.model_dump_json(exclude={"duration_ms"})))

        return finish("max_model_turns_exceeded", cfg.max_model_turns,
                      error=f"Model-turn limit of {cfg.max_model_turns} reached without a final answer.")
