"""
Evaluation wrapper around orena-focus Evaluator.
Adds logging and per-capability breakdown display.
"""
import os
import pandas as pd
from focus import Evaluator, Response


def run_evaluation(requests, references, raw_answers: list[str], output_dir: str = None):
    """
    requests: list of FocusDataset request objects
    references: list of FocusDataset reference objects
    raw_answers: list of string answers from your model (same order)
    """
    responses = [
        Response(qID=req.qID, content=ans)
        for req, ans in zip(requests, raw_answers)
    ]

    results_df, summary_df = Evaluator().run(
        requests=requests,
        references=references,
        responses=responses,
    )

    print("\n=== Summary ===")
    print(summary_df.to_string())

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
        summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
        print(f"\nResults saved to {output_dir}/")

    return results_df, summary_df


def show_failures(results_df: pd.DataFrame, n: int = 10):
    """Quick look at worst predictions — essential for error analysis."""
    if "correct" in results_df.columns:
        failures = results_df[results_df["correct"] == False].head(n)
    else:
        failures = results_df.head(n)
    print(failures[["question", "prediction", "reference", "capability"]].to_string())
    return failures
