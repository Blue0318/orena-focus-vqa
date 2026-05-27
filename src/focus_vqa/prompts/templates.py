"""
Prompt templates — built from actual dataset inspection.

Key findings from dataset analysis:
- fo_class answers use exact class names, comma-separated alphabetically (mostly)
- Spatial answers follow: "1. Class: position 2. Class: position"
- Multi-label ordering is inconsistent in ground truth — flag for forum
- 'not safe to answer' is a valid output for ambiguous frames
- Dataset typo: 'silcon loop' exists — normalization needed in postprocessing
"""

FO_CLASSES = [
    "Clip", "External drain", "Needle",
    "Silicone loop", "Specimen", "Specimen bag", "Sponge",
]

SPATIAL_POSITIONS = ["top/left", "top/right", "bottom/left", "bottom/right"]

SAFETY_ANSWERS = [
    "not safe to answer - frame question",
    "frame question - not safe to answer",
]

FO_CLASS_HINT = (
    "Valid foreign object class names (exact spelling and capitalisation): "
    + ", ".join(FO_CLASSES)
    + ". For multiple objects, list class names separated by ', ' in alphabetical order."
)

SPATIAL_HINT = (
    "Format: '1. ClassName: position 2. ClassName: position ...'\n"
    f"Valid positions: {', '.join(SPATIAL_POSITIONS)}.\n"
    f"Valid class names: {', '.join(FO_CLASSES)}."
)

TYPO_MAP = {
    "silcon loop": "Silicone loop",
    "silicon loop": "Silicone loop",
    "surgical clip": "Clip",
    "clips": "Clip",
    "sponges": "Sponge",
}


def build_prompt(request, system_prompt: str = "", use_metadata: bool = True) -> str:
    parts = []

    if use_metadata:
        meta_parts = []
        if hasattr(request, "procedure_type") and request.procedure_type:
            meta_parts.append(f"Procedure: {request.procedure_type}")
        if hasattr(request, "timestamp") and request.timestamp:
            meta_parts.append(f"Timestamp: {request.timestamp}")
        if meta_parts:
            parts.append("[" + " | ".join(meta_parts) + "]")

    parts.append(request.question)

    fmt = str(request.format.type).lower() if hasattr(request, "format") else ""

    if fmt == "binary":
        parts.append("Answer with yes or no only.")
    elif fmt == "number":
        parts.append("Answer with a single non-negative integer only.")
    elif fmt == "fo_class":
        parts.append(FO_CLASS_HINT)
    elif fmt == "multiple_choice":
        parts.append("Answer with exactly one of the provided options.")
    elif fmt == "open_ended":
        q_lower = request.question.lower()
        if any(w in q_lower for w in ["where", "location", "position", "quadrant", "located"]):
            parts.append(SPATIAL_HINT)
        else:
            parts.append(
                f"Answer concisely. If ambiguous, respond with: '{SAFETY_ANSWERS[0]}'"
            )

    return "\n".join(parts)


def normalize_fo_class(raw: str) -> str:
    corrected = raw.strip()
    for wrong, right in TYPO_MAP.items():
        corrected = corrected.replace(wrong, right)
    if "," in corrected:
        parts = sorted([p.strip() for p in corrected.split(",")])
        corrected = ", ".join(parts)
    return corrected


FEW_SHOT_EXAMPLES = {
    "binary": [
        {"question": "Is a sponge visible?", "answer": "yes"},
        {"question": "Is a needle present?", "answer": "no"},
    ],
    "number": [{"question": "How many clips are visible?", "answer": "2"}],
    "fo_class": [{"question": "What foreign object is visible?", "answer": "Sponge"}],
}


def build_few_shot_prefix(format_type: str) -> str:
    examples = FEW_SHOT_EXAMPLES.get(format_type.lower(), [])
    if not examples:
        return ""
    lines = ["Examples:"]
    for ex in examples:
        lines.append(f"Q: {ex['question']}\nA: {ex['answer']}")
    lines.append("Now answer:")
    return "\n".join(lines)
