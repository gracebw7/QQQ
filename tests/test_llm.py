"""StructuredLLM: LangChain structured output, validation, retries and errors.

Uses a fake chat model that emits real tool-call messages, so LangChain's own
with_structured_output parsing (and our Pydantic validators) run for real.
"""

import itertools

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm import LLMError, StructuredLLM, get_llm
from app.schemas import RouterDecision, RouterStatus


class FakeToolChatModel(GenericFakeChatModel):
    """GenericFakeChatModel that accepts bind_tools and records every input."""

    calls: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return super()._generate(messages, *args, **kwargs)


def tool_reply(args: dict, name: str = "RouterDecision") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])


def fake_model(*replies) -> FakeToolChatModel:
    return FakeToolChatModel(messages=iter(replies), calls=[])


DISTILLED = {
    "status": "distilled",
    "distilled_question": "How much memory does LoRA save?",
    "reason": "single question",
}


def generate(llm: StructuredLLM, user: str = "hi"):
    return llm.generate(system="SYS", user=user, schema=RouterDecision)


def test_generate_returns_validated_schema_instance():
    model = fake_model(tool_reply(DISTILLED))
    result = generate(StructuredLLM(model))
    assert isinstance(result, RouterDecision)
    assert result.status is RouterStatus.DISTILLED
    assert result.distilled_question == "How much memory does LoRA save?"


def test_generate_sends_system_then_user_message():
    model = fake_model(tool_reply(DISTILLED))
    generate(StructuredLLM(model), user="what is LoRA?")
    [messages] = model.calls
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage]
    assert messages[0].content == "SYS"
    assert messages[1].content == "what is LoRA?"


@pytest.mark.parametrize("user", ["{question}", "use {{braces}} and {0}", "}{"])
def test_user_text_with_braces_is_passed_verbatim(user):
    """Braces must not be treated as prompt-template variables."""
    model = fake_model(tool_reply(DISTILLED))
    generate(StructuredLLM(model), user=user)
    assert model.calls[0][1].content == user


def test_generate_strips_whitespace_in_model_output():
    model = fake_model(tool_reply({**DISTILLED, "distilled_question": "  Why?  \n"}))
    assert generate(StructuredLLM(model)).distilled_question == "Why?"


@pytest.mark.parametrize(
    "bad_args",
    [
        {"status": "distilled"},  # missing required question
        {"status": "distilled", "distilled_question": "   "},  # blank after strip
        {"status": "needs_reframe"},  # missing reframe message
        {"status": "maybe", "distilled_question": "Why?"},  # unknown status
    ],
)
def test_invalid_output_is_retried_then_succeeds(bad_args):
    model = fake_model(tool_reply(bad_args), tool_reply(DISTILLED))
    result = generate(StructuredLLM(model, max_attempts=2))
    assert result.status is RouterStatus.DISTILLED
    assert len(model.calls) == 2


def test_invalid_output_on_every_attempt_raises_llm_error():
    model = fake_model(*itertools.repeat(tool_reply({"status": "distilled"}), 5))
    with pytest.raises(LLMError, match="after 3 attempts"):
        generate(StructuredLLM(model, max_attempts=3))
    assert len(model.calls) == 3  # no extra calls beyond the limit


def test_reply_without_tool_call_is_treated_as_invalid():
    """A plain-text answer (model ignored the tool) must not pass as a result."""
    model = fake_model(AIMessage(content="Sure! Here's the question..."), tool_reply(DISTILLED))
    assert generate(StructuredLLM(model)).status is RouterStatus.DISTILLED
    assert len(model.calls) == 2


def test_provider_error_is_wrapped_and_not_retried():
    class Boom(FakeToolChatModel):
        def _generate(self, messages, *args, **kwargs):
            self.calls.append(messages)
            raise ConnectionError("network down")

    model = Boom(messages=iter([]), calls=[])
    with pytest.raises(LLMError) as info:
        generate(StructuredLLM(model))
    assert isinstance(info.value.__cause__, ConnectionError)
    assert len(model.calls) == 1


def test_to_schema_rejects_result_flagged_with_parsing_error():
    """Never trust `parsed` when LangChain also reports a parsing error."""
    from app.llm import _to_schema

    ok = RouterDecision(**DISTILLED)
    with pytest.raises(ValueError, match="parsing error"):
        _to_schema({"raw": None, "parsed": ok, "parsing_error": Exception("bad")}, RouterDecision)
    with pytest.raises(ValueError, match="no structured output"):
        _to_schema({"raw": None, "parsed": None, "parsing_error": None}, RouterDecision)


def test_to_schema_validates_dict_output():
    """Providers using JSON mode return dicts; they must still be validated."""
    from app.llm import _to_schema

    assert _to_schema(DISTILLED, RouterDecision).status is RouterStatus.DISTILLED
    with pytest.raises(ValueError):
        _to_schema({"status": "distilled"}, RouterDecision)


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        StructuredLLM(fake_model(), max_attempts=0)


def test_get_llm_uses_env_model_and_deterministic_settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("QQQ_LLM_MODEL", "anthropic:claude-test-model")
    llm = get_llm()
    assert isinstance(llm, StructuredLLM)
    assert llm.chat_model.model == "claude-test-model"
    assert llm.chat_model.temperature == 0


def test_get_llm_explicit_model_overrides_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("QQQ_LLM_MODEL", "anthropic:claude-from-env")
    assert get_llm("anthropic:claude-explicit").chat_model.model == "claude-explicit"
