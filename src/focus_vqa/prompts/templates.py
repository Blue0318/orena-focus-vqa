"""
Prompt templates.
Centralising prompts here means you can A/B test them by changing one file,
not hunting through model code.
"""


def build_prompt(request, system_prompt: str = "", use_metadata: bool = True) -> str:
    """
    Build the text portion of the prompt for a FRAME request.
    The image is handled separately in the model wrapper.
    """
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

    # Answer format hint — reduces hallucination on constrained formats
    if hasattr(request, "answer_format"):
        fmt = request.answer_format
        if str(fmt).lower() == "binary":
            parts.append("Answer with yes or no only.")
        elif str(fmt).lower() == "number":
            parts.append("Answer with a number only.")
        elif str(fmt).lower() == "percentage":
            parts.append("Answer with a percentage only (e.g. 50%).")
        elif str(fmt).lower() == "multiplechoice":
            parts.append("Answer with one of the provided options only.")

    return "\n".join(parts)


# --- Few-shot templates (expand as you gather good examples) ---

FEW_SHOT_EXAMPLES = {
    "binary": [
        {
            "question": "Is a sponge visible in this image?",
            "answer": "yes",
        }
    ],
    "number": [
        {
            "question": "How many clips are visible?",
            "answer": "2",
        }
    ],
}


def build_few_shot_prefix(format_type: str) -> str:
    examples = FEW_SHOT_EXAMPLES.get(format_type.lower(), [])
    if not examples:
        return ""
    lines = ["Here are example question-answer pairs:"]
    for ex in examples:
        lines.append(f"Q: {ex['question']}")
        lines.append(f"A: {ex['answer']}")
    lines.append("Now answer the following:")
    return "\n".join(lines)
