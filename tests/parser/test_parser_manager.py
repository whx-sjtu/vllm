# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.parser.harmony import HarmonyParser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser


@pytest.fixture(autouse=True)
def restore_harmony_delegates():
    # get_parser assigns the delegates onto the HarmonyParser class itself.
    reasoning_cls = HarmonyParser.reasoning_parser_cls
    tool_cls = HarmonyParser.tool_parser_cls
    yield
    HarmonyParser.reasoning_parser_cls = reasoning_cls
    HarmonyParser.tool_parser_cls = tool_cls


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"reasoning_parser_name": "openai_gptoss"},
        {"tool_parser_name": "openai", "enable_auto_tools": True},
        {"tool_parser_name": "openai", "enable_auto_tools": False},
        {
            "reasoning_parser_name": "openai_gptoss",
            "tool_parser_name": "openai",
            "enable_auto_tools": True,
        },
    ],
)
def test_harmony_model_always_gets_harmony_parser(kwargs):
    """Harmony is the model's native format, not a user opt-in.

    Serving a Harmony model without --reasoning-parser/--tool-call-parser used
    to yield no parser at all, so the harmony channels were never split out of
    the raw output.
    """
    assert ParserManager.get_parser(is_harmony=True, **kwargs) is HarmonyParser


def test_harmony_parser_delegates_follow_configuration():
    ParserManager.get_parser(is_harmony=True)
    assert HarmonyParser.reasoning_parser_cls is None
    assert HarmonyParser.tool_parser_cls is None

    ParserManager.get_parser(
        reasoning_parser_name="openai_gptoss",
        tool_parser_name="openai",
        enable_auto_tools=True,
        is_harmony=True,
    )
    assert HarmonyParser.reasoning_parser_cls is GptOssReasoningParser
    assert HarmonyParser.tool_parser_cls is GptOssToolParser


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"tool_parser_name": "hermes"},
        {"tool_parser_name": "hermes", "enable_auto_tools": False},
    ],
)
def test_no_parser_without_usable_configuration(kwargs):
    assert ParserManager.get_parser(**kwargs) is None
