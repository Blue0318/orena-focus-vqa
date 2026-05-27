"""
Submission entrypoint — called by the challenge evaluation runner.
Do NOT change the interface contract; only change what's inside predict().

The challenge runner will call this script and pass input via stdin or file
(exact format TBD — update once submission template is released).
"""
import sys
import json

# TODO: update import paths once submission template spec is released
from focus_vqa.models.qwen_vl import QwenVLModel


def predict(request_data: dict) -> str:
    """
    Given a request dict from the challenge runner, return a string answer.
    This is a placeholder — adapt once the official submission template is out.
    """
    raise NotImplementedError("Update once submission template is released")


if __name__ == "__main__":
    # Placeholder main — will be replaced once submission spec is published
    input_data = json.load(sys.stdin)
    answer = predict(input_data)
    print(answer)
