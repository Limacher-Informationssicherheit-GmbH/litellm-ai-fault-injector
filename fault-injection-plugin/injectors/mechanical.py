# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic, seed-driven injectors — no extra LLM call.

These are fully reproducible from the RNG seed, so they are the injector set
used in ``deterministic`` mode and in unit tests.

- ``BadCodeInjector``: finds a fenced code block and introduces one subtle bug.
  Python blocks are mutated via ``ast`` so the swap targets a real operator;
  other languages fall back to a conservative token-level swap, and if nothing
  safe is found the injector declines (returns ``None``).
- ``FakeSourceInjector``: appends a plausible-but-fabricated citation.
"""

from __future__ import annotations

import ast
import random
import re
from typing import List, Optional, Tuple

from .base import InjectionRecord, InjectionResult

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

# operator text swaps used for the non-Python fallback and as AST targets
_BINOP_SWAP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
}
_OP_TEXT = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//",
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=",
}


def _find_code_blocks(content: str) -> List[re.Match]:
    return list(_FENCE_RE.finditer(content))


def _line_starts(code: str) -> List[int]:
    """Absolute char offset at which each 1-based source line begins."""
    starts = [0]
    for i, ch in enumerate(code):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _find_op_outside_comments(gap: str, op_text: str) -> int:
    """Index of ``op_text`` in ``gap``, skipping any ``#`` comment spans.

    The gap (text strictly between two operands) can only contain whitespace,
    line continuations, and comments — never a string literal (that would be an
    operand). So skipping ``#``-to-end-of-line is enough to avoid matching an
    operator character that merely appears inside a comment.
    """
    i, n = 0, len(gap)
    while i < n:
        ch = gap[i]
        if ch == "#":
            nl = gap.find("\n", i)
            if nl == -1:
                return -1  # comment runs to the end; no operator after it
            i = nl + 1
            continue
        if gap.startswith(op_text, i):
            return i
        i += 1
    return -1


def _abs_offset(line_starts: List[int], lineno: int, col: int) -> int:
    """Map a 1-based (lineno, col) AST position to an absolute char offset.

    ``col_offset`` is a UTF-8 byte offset; for the ASCII operators/identifiers
    this targets it equals the char offset. Non-ASCII before the operator could
    skew it, but the caller verifies the operator text is actually present in
    the computed gap and otherwise declines — so a bad offset never corrupts.
    """
    return line_starts[lineno - 1] + col


class BadCodeInjector:
    error_type = "bad_code"

    def applies(self, content: str) -> bool:
        return bool(_find_code_blocks(content))

    async def inject(
        self, content: str, rng: random.Random
    ) -> Optional[InjectionResult]:
        blocks = _find_code_blocks(content)
        if not blocks:
            return None
        # pick a block deterministically from the seeded rng
        match = rng.choice(blocks)
        lang = (match.group(1) or "").strip().lower()
        code = match.group(2)

        out = self._mutate(code, lang, rng)
        if out is None:
            return None  # nothing safe to change -> decline
        mutated, detail = out

        new_content = content[: match.start(2)] + mutated + content[match.end(2) :]
        return InjectionResult(
            content=new_content,
            record=InjectionRecord(
                error_type=self.error_type,
                detail=detail,
                meta={"lang": lang or "unknown"},
            ),
        )

    def _mutate(
        self, code: str, lang: str, rng: random.Random
    ) -> Optional[Tuple[str, str]]:
        if lang in ("", "python", "py"):
            out = self._mutate_python(code, rng)
            if out is not None:
                return out
        return self._mutate_tokens(code, rng)

    def _mutate_python(
        self, code: str, rng: random.Random
    ) -> Optional[Tuple[str, str]]:
        # Use the AST only to LOCATE one operator, then splice that single
        # symbol into the ORIGINAL source. We deliberately do NOT round-trip
        # through ast.unparse: that would reformat the whole block, drop the
        # trailing newline (gluing on the closing fence), and strip comments —
        # turning a "subtle" fault into an obvious one.
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        line_starts = _line_starts(code)
        targets = []  # (left_node, right_node, original_op_type)
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and type(node.op) in _BINOP_SWAP:
                targets.append((node.left, node.right, type(node.op)))
            elif (
                isinstance(node, ast.Compare)
                and node.ops
                and type(node.ops[0]) in _BINOP_SWAP
            ):
                targets.append((node.left, node.comparators[0], type(node.ops[0])))
        if not targets:
            return None

        rng.shuffle(targets)
        for left, right, orig_t in targets:
            try:
                l_end = _abs_offset(line_starts, left.end_lineno, left.end_col_offset)
                r_start = _abs_offset(line_starts, right.lineno, right.col_offset)
            except (AttributeError, IndexError, TypeError):
                continue
            if not 0 <= l_end <= r_start <= len(code):
                continue
            gap = code[l_end:r_start]  # operator + whitespace/comments only
            op_text = _OP_TEXT[orig_t]
            # Locate the operator OUTSIDE any `#` comment. In a parenthesized
            # multi-line expression the gap can contain a comment whose text
            # holds the operator char (e.g. "# a+b"); a naive find() would swap
            # that instead, mangling the comment and shipping a phantom
            # injection that changes nothing but logs as "injected".
            idx = _find_op_outside_comments(gap, op_text)
            if idx == -1:
                continue  # couldn't locate the real operator token -> try next
            new_op = _OP_TEXT[_BINOP_SWAP[orig_t]]
            new_gap = gap[:idx] + new_op + gap[idx + len(op_text) :]
            new_code = code[:l_end] + new_gap + code[r_start:]
            # Defense-in-depth against a bad offset (e.g. non-ASCII byte/char
            # skew): only ship if the result is still valid Python.
            try:
                ast.parse(new_code)
            except SyntaxError:
                continue
            detail = f"swapped {op_text} -> {new_op} in python block"
            return new_code, detail
        return None

    def _mutate_tokens(
        self, code: str, rng: random.Random
    ) -> Optional[Tuple[str, str]]:
        # conservative surface swaps for non-Python code; require word/space
        # boundaries so we don't corrupt operators inside identifiers.
        swaps = [(" + ", " - "), (" - ", " + "), (" <= ", " < "), (" >= ", " > ")]
        rng.shuffle(swaps)
        for a, b in swaps:
            if a in code:
                mutated = code.replace(a, b, 1)
                return mutated, f"swapped '{a.strip()}' -> '{b.strip()}' in code"
        return None


class FakeSourceInjector:
    error_type = "fake_source"

    _AUTHORS = ["Hartmann", "Novak", "Feldman", "Okonkwo", "Reyes", "Bianchi"]
    _JOURNALS = [
        "Journal of Applied Systems",
        "Proceedings of the Intl. Conf. on Data Integrity",
        "Review of Computational Methods",
    ]

    def applies(self, content: str) -> bool:
        # a fabricated citation is meaningful on any prose of reasonable length
        return len(content.strip()) >= 40

    async def inject(
        self, content: str, rng: random.Random
    ) -> Optional[InjectionResult]:
        author = rng.choice(self._AUTHORS)
        journal = rng.choice(self._JOURNALS)
        year = rng.randint(2015, 2024)
        vol = rng.randint(3, 41)
        page = rng.randint(10, 320)
        citation = (
            f"{author} et al. ({year}), *{journal}*, {vol}({rng.randint(1,4)}), "
            f"pp. {page}-{page + rng.randint(4, 18)}."
        )
        new_content = (
            content.rstrip()
            + f"\n\nSource: {citation}"
        )
        return InjectionResult(
            content=new_content,
            record=InjectionRecord(
                error_type=self.error_type,
                detail="appended fabricated citation",
                meta={"citation": citation},
            ),
        )
