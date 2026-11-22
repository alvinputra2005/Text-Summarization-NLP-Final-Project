from html import escape
import re


IDENTIFIER = r"[A-Za-z_]\w*"
QUALIFIED = rf"{IDENTIFIER}(?:\.{IDENTIFIER})*"
CODE_TOKEN = (
    rf"\b{QUALIFIED}\(\)|\b{QUALIFIED}(?=\()"
    r"|\b[A-Za-z]\w*_\w+\b"
    r"|\b(?:Python|pandas|CSV|DataFrame|input|print|int|float|bool|str|string|integer|boolean|type|True|False|None|"
    r"dropna|fillna|isna|isnull|head|describe|subset|axis|inplace|index|dtype|not|or|and|XOR)\b|\^"
)
DECLARED = re.compile(rf"\b(?:variabel\s+({IDENTIFIER})|({IDENTIFIER})\s*=(?!=))")


def code_matches(text: str):
    names = sorted({match[1] or match[2] for match in DECLARED.finditer(text)})
    variables = [rf"\b{re.escape(name)}\b" for name in names]
    return re.finditer("|".join([CODE_TOKEN, *variables]), text)


def technical_terms(text: str) -> set[str]:
    return {match[0].casefold().removesuffix("()") for match in code_matches(text)}


def highlight_code(text: str) -> str:
    chunks = []
    end = 0
    for match in code_matches(text):
        chunks.extend((escape(text[end:match.start()]), f"<code>{escape(match[0])}</code>"))
        end = match.end()
    return "".join(chunks) + escape(text[end:])
