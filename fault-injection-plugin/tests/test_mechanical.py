# SPDX-License-Identifier: AGPL-3.0-or-later
import random

import pytest

from injectors.mechanical import BadCodeInjector, FakeSourceInjector

PY_BLOCK = "Here is code:\n\n```python\ndef add(a, b):\n    return a + b\n```\n"
PROSE = "The capital of France is Paris and the Eiffel Tower opened in 1889."


def rng(seed=1):
    return random.Random(seed)


async def test_bad_code_applies_only_with_code():
    inj = BadCodeInjector()
    assert inj.applies(PY_BLOCK) is True
    assert inj.applies(PROSE) is False


async def test_bad_code_swaps_python_operator():
    inj = BadCodeInjector()
    result = await inj.inject(PY_BLOCK, rng(42))
    assert result is not None
    # the '+' inside the code block should have flipped to '-'
    assert "return a - b" in result.content
    assert result.record.error_type == "bad_code"
    assert "python" in result.record.detail


async def test_bad_code_preserves_fence_and_comments():
    # regression for the ast.unparse round-trip that stripped comments and
    # glued the closing fence onto the last code line.
    code = (
        "```python\n"
        "def f(a, b):\n"
        "    # add the two operands\n"
        "    return a + b  # the sum\n"
        "```"
    )
    result = await BadCodeInjector().inject(code, rng(3))
    assert result is not None
    c = result.content
    assert "return a - b" in c            # operator flipped
    assert "# add the two operands" in c  # comment preserved
    assert "# the sum" in c               # trailing comment preserved
    assert "\n```" in c                   # closing fence still on its own line
    assert "b```" not in c                # fence not glued to the code


async def test_bad_code_ignores_operator_inside_comment():
    # regression: a '+' inside an inline comment must NOT be swapped (that would
    # mangle the comment and log a phantom injection that changes nothing). The
    # real operator on the continuation line is the one that flips.
    code = (
        "```python\n"
        "z = (a  # note: a+b appears here\n"
        "     + b)\n"
        "```"
    )
    result = await BadCodeInjector().inject(code, rng(2))
    assert result is not None
    assert "# note: a+b appears here" in result.content  # comment untouched
    assert "- b)" in result.content                       # real operator swapped


async def test_bad_code_swaps_operator_without_spaces():
    # the AST-locate-then-splice approach handles 'a+b' (no whitespace), which
    # the token fallback could not.
    result = await BadCodeInjector().inject("```python\nr = a+b\n```", rng(1))
    assert result is not None
    assert "a-b" in result.content


async def test_bad_code_deterministic_same_seed():
    inj = BadCodeInjector()
    a = await inj.inject(PY_BLOCK, rng(7))
    b = await inj.inject(PY_BLOCK, rng(7))
    assert a.content == b.content


async def test_bad_code_declines_without_operators():
    inj = BadCodeInjector()
    code = "```python\nx = 1\nprint(x)\n```"
    assert await inj.inject(code, rng(1)) is None


async def test_fake_source_appends_citation():
    inj = FakeSourceInjector()
    assert inj.applies(PROSE) is True
    result = await inj.inject(PROSE, rng(3))
    assert result is not None
    assert "Source:" in result.content
    assert result.content.startswith(PROSE)
    assert result.record.meta["citation"]


async def test_fake_source_declines_on_tiny_input():
    inj = FakeSourceInjector()
    assert inj.applies("hi") is False
