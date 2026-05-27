"""
Sanity tests — run with: pytest tests/
"""
import pytest


def test_prompt_builder_no_metadata():
    from focus_vqa.prompts.templates import build_prompt

    class FakeRequest:
        question = "How many sponges are visible?"
        procedure_type = None
        timestamp = None
        answer_format = "number"

    prompt = build_prompt(FakeRequest(), system_prompt="", use_metadata=False)
    assert "How many sponges" in prompt
    assert "number" in prompt.lower() or "Number" in prompt


def test_prompt_builder_with_metadata():
    from focus_vqa.prompts.templates import build_prompt

    class FakeRequest:
        question = "Is a clip visible?"
        procedure_type = "colorectal"
        timestamp = "00:12:34"
        answer_format = "binary"

    prompt = build_prompt(FakeRequest(), system_prompt="", use_metadata=True)
    assert "colorectal" in prompt
    assert "00:12:34" in prompt
    assert "yes or no" in prompt.lower()
