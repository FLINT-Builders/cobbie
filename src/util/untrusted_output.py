"""Fence and sanitize untrusted text before it is placed in an LLM prompt.

Execution stdout and IFC model summaries carry text written by whoever authored
the building model (element names, descriptions, property values). That text is
data. These helpers wrap it in a fence the text itself cannot open or close and
remove lines that read as instructions to the model. Prompts that render the
fenced text must also state, outside the fence, that it is evidence only.
"""

import re
import unicodedata
from typing import List, Tuple

EXECUTION_OUTPUT_BEGIN = "<<<UNTRUSTED_EXECUTION_OUTPUT>>>"
EXECUTION_OUTPUT_END = "<<<END_UNTRUSTED_EXECUTION_OUTPUT>>>"
MODEL_SUMMARY_BEGIN = "<<<UNTRUSTED_MODEL_SUMMARY>>>"
MODEL_SUMMARY_END = "<<<END_UNTRUSTED_MODEL_SUMMARY>>>"

REMOVED_MARKER = "[fence marker removed]"
REMOVED_INSTRUCTION = "[instruction-like text removed]"

# Latin look-alikes used to dodge keyword matching. Applied only to the view
# used for matching; the returned text keeps the original characters.
_CONFUSABLES = str.maketrans(
    {
        "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h",
        "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "ѕ": "s",
        "і": "i", "ї": "i", "ј": "j", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ɡ": "g",
        "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v",
        "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ω": "w",
        "‹": "<", "«": "<", "⟨": "<", "〈": "<", "❮": "<",
        "›": ">", "»": ">", "⟩": ">", "〉": ">", "❯": ">",
    }
)

_FILLER = (
    r"(?:(?:all|any|the|these|those|your|my|this|that|of|previous|prior|above|"
    r"preceding|earlier|following|other|existing|script|scripts|code|execution|"
    r"response|quality|and|or)\s)*"
)

_BRACKET_RUN = re.compile(r"<{2,}|>{2,}")
_MARKER_WORD = re.compile(
    r"(?:end[\W_]*)?untrusted(?:[\W_]*(?:execution|model))?(?:[\W_]*(?:output|summary))?"
)

# Matched against the text with every run of non-alphanumerics collapsed to a
# single space, so separators and line breaks between words do not matter.
_WORD_DIRECTIVES = [
    re.compile(
        r"\b(?:ignore|disregard|forget|override|overrule|bypass|discard)\s"
        + _FILLER
        + r"(?:instructions?|rules?|criteria|criterion|prompts?|outputs?|results?|"
        r"context|guidelines?|directions?|questions?|task|system|messages?|"
        r"evidence|data|above|everything|constraints?)\b"
    ),
    re.compile(
        r"\b(?:answer|reply|response)(?:\s(?:field|text|value|key|string))?\s"
        r"(?:must|should|shall|will|has to|needs to|is to|ought to)\b"
    ),
    re.compile(
        r"\b(?:correct|real|true|actual|right|only|expected)\s(?:answer|response|reply)\s"
        r"(?:is|was|should|must)\b"
    ),
    re.compile(r"\bfinal\s(?:answer|response|output)\b"),
    re.compile(r"\banswer\sfield\b"),
    re.compile(
        r"\b(?:answer|respond|reply|say)\s(?:only\s|with\s|using\s|just\s)*"
        r"(?:exactly|only|verbatim|precisely|nothing but|nothing else|the string|"
        r"the word|the phrase|the text)\b"
    ),
    re.compile(r"\b(?:respond|reply|answer)\s(?:with|using)\b"),
    re.compile(
        r"\b(?:answer|respond|reply|say|output|state)\b(?:\s[a-z0-9]+){0,8}\s"
        r"(?:nothing else|nothing more|verbatim|word for word)\b"
    ),
    re.compile(r"\b(?:just|only|simply|always)\s(?:answer|respond|reply|say)\b"),
    re.compile(r"\bas\s(?:the|your)\s(?:final\s)?(?:answer|response|reply)\b"),
    re.compile(r"\btell\sthe\suser\b"),
    re.compile(r"\binstead\s(?:answer|respond|reply|say|output|return)\b"),
    re.compile(
        r"\byou\s(?:must|should|shall|will|are to|need to|have to|are required to|"
        r"are instructed to)\s(?:now\s|instead\s|only\s|always\s)*"
        r"(?:answer|respond|reply|say|output|ignore|disregard|forget|claim|obey|pretend)\b"
    ),
    re.compile(
        r"\b(?:new|updated|revised|real|actual|hidden|secret|override|system|developer|admin)\s"
        r"(?:instructions?|directives?|prompt)\b"
    ),
    re.compile(r"\b(?:system|developer|assistant)\s(?:prompt|message|instructions?)\b"),
    re.compile(r"\bfrom\snow\son\b"),
]

# Matched against the text with all non-alphanumerics removed, which catches
# letter-spaced or punctuated spellings of the most direct phrasings.
_SQUASHED_DIRECTIVES = [
    re.compile(
        r"(?:ignore|disregard)(?:the|all|any|previous|prior|above|your|script|code|"
        r"response|quality|and)*(?:instructions?|output|criteria|rules?|prompt|context)"
    ),
    re.compile(r"answer(?:field)?(?:must|should|shall)(?:be|say|read|contain|equal)"),
    re.compile(r"finalanswer"),
]

# Matched against the case-folded text with punctuation intact.
_CHAR_DIRECTIVES = [
    re.compile(r"<\|[^|>\n]{1,40}\|>"),
    re.compile(r"\[/?inst\]|<{1,2}/?sys>{1,2}|_\.role\s*\("),
    re.compile(r"[\[<]\s*/?\s*(?:system|assistant|developer|user)\s*[\]>]"),
    re.compile(r"(?m)^\s*(?:assistant|developer)\s*:"),
    re.compile(r"[\"']\s*(?:answer|thoughts)\s*[\"']\s*:"),
]


def _char_view(text: str) -> Tuple[str, List[int]]:
    """Normalized, case-folded copy of ``text`` with a map back to source indices."""
    chars: List[str] = []
    index_map: List[int] = []
    for i, ch in enumerate(text):
        if unicodedata.category(ch) == "Cf":
            continue
        normalized = unicodedata.normalize("NFKC", ch).casefold().translate(_CONFUSABLES)
        for n in normalized:
            chars.append(n)
            index_map.append(i)
    return "".join(chars), index_map


def _project(view: str, index_map: List[int], keep_separator: bool) -> Tuple[str, List[int]]:
    """Collapse (or drop) runs of non-alphanumerics in a view, keeping the index map."""
    chars: List[str] = []
    out_map: List[int] = []
    in_separator = False
    for ch, src in zip(view, index_map):
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            chars.append(ch)
            out_map.append(src)
            in_separator = False
        elif keep_separator and not in_separator and chars:
            chars.append(" ")
            out_map.append(src)
            in_separator = True
    return "".join(chars), out_map


def _spans(patterns, view: str, index_map: List[int]) -> List[Tuple[int, int]]:
    spans = []
    for pattern in patterns:
        for match in pattern.finditer(view):
            if match.end() <= match.start():
                continue
            spans.append((index_map[match.start()], index_map[match.end() - 1] + 1))
    return spans


def neutralize_fence_markers(text: str) -> str:
    """Remove anything that could be read as opening or closing an untrusted fence."""
    view, index_map = _char_view(text)
    replacements = [
        (start, end, REMOVED_MARKER) for start, end in _spans([_MARKER_WORD], view, index_map)
    ]
    for match in _BRACKET_RUN.finditer(view):
        start = index_map[match.start()]
        end = index_map[match.end() - 1] + 1
        replacements.append((start, end, match.group()[0]))
    if not replacements:
        return text
    replacements.sort()
    out = []
    cursor = 0
    for start, end, replacement in replacements:
        if start < cursor:
            continue
        out.append(text[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def strip_instruction_lines(text: str) -> str:
    """Replace every line that contains instruction-like text with a placeholder."""
    view, index_map = _char_view(text)
    word_view, word_map = _project(view, index_map, keep_separator=True)
    squashed_view, squashed_map = _project(view, index_map, keep_separator=False)
    spans = (
        _spans(_CHAR_DIRECTIVES, view, index_map)
        + _spans(_WORD_DIRECTIVES, word_view, word_map)
        + _spans(_SQUASHED_DIRECTIVES, squashed_view, squashed_map)
    )
    if not spans:
        return text

    lines = text.split("\n")
    line_starts = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1

    def line_of(position: int) -> int:
        low, high = 0, len(line_starts) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if line_starts[mid] <= position:
                low = mid
            else:
                high = mid - 1
        return low

    flagged = set()
    for start, end in spans:
        flagged.update(range(line_of(start), line_of(end - 1) + 1))

    out: List[str] = []
    for number, line in enumerate(lines):
        if number not in flagged:
            out.append(line)
        elif not out or out[-1] != REMOVED_INSTRUCTION:
            out.append(REMOVED_INSTRUCTION)
    return "\n".join(out)


def sanitize_untrusted_text(text: str) -> str:
    """Neutralize fence markers, then drop instruction-like lines."""
    return strip_instruction_lines(neutralize_fence_markers(text))


def fence_execution_output(text: str) -> str:
    """Sanitize captured execution output and wrap it in the untrusted-output fence."""
    return f"{EXECUTION_OUTPUT_BEGIN}\n{sanitize_untrusted_text(text)}\n{EXECUTION_OUTPUT_END}"


def fence_model_summary(text: str) -> str:
    """Sanitize an IFC model summary and wrap it in the untrusted-summary fence."""
    return f"{MODEL_SUMMARY_BEGIN}\n{sanitize_untrusted_text(text)}\n{MODEL_SUMMARY_END}"
