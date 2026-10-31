"""Recognize escaped literal directory components in bounded glob patterns."""


def literal_component(component: str) -> str | None:
    """Decode glob.escape spellings; return None for an actual wildcard."""
    result = ""
    index = 0
    while index < len(component):
        escape = component[index : index + 3]
        if escape in {"[[]", "[*]", "[?]"}:
            result += escape[1]
            index += 3
        elif component[index] in "*?[":
            return None
        else:
            result += component[index]
            index += 1
    return result
