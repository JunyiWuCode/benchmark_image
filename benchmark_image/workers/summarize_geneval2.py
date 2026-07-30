#!/usr/bin/env python
"""Persist structured GenEval2 metrics from the official per-atom score lists."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import gmean


SKILLS = ("object", "attribute", "count", "position", "verb")
ATOMICITIES = tuple(range(3, 11))
OFFICIAL_PROMPT_COUNT = 800
OFFICIAL_QUESTION_COUNT = 6012


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty score list.")
    return sum(values) / len(values)


def summarize(
    benchmark_rows: list[dict],
    score_lists: list[list[float]],
    *,
    allow_partial: bool = False,
) -> dict:
    if len(benchmark_rows) != len(score_lists):
        raise ValueError(
            f"GenEval2 prompt/score count mismatch: {len(benchmark_rows)} vs {len(score_lists)}"
        )

    skill_scores: dict[str, list[float]] = defaultdict(list)
    atomicity_scores: dict[int, list[float]] = defaultdict(list)
    prompt_am: list[float] = []
    prompt_gm: list[float] = []
    question_count = 0

    for index, (record, scores) in enumerate(zip(benchmark_rows, score_lists)):
        skills = list(record["skills"])
        vqa_list = list(record["vqa_list"])
        if len(scores) != len(skills) or len(scores) != len(vqa_list):
            raise ValueError(
                f"GenEval2 row {index} is misaligned: scores={len(scores)} "
                f"skills={len(skills)} vqa={len(vqa_list)}"
            )
        numeric_scores = [float(score) for score in scores]
        # The official evaluator sums probabilities for several answer spellings.
        # Duplicate token ids can put that sum a few ulps above one.
        if any(
            not math.isfinite(score) or score < 0.0 or score > 1.0 + 1e-5
            for score in numeric_scores
        ):
            raise ValueError(f"GenEval2 row {index} contains an invalid probability.")

        prompt_am.append(_mean(numeric_scores))
        prompt_gm.append(float(gmean(numeric_scores)))
        question_count += len(numeric_scores)
        for skill, score in zip(skills, numeric_scores):
            if skill not in SKILLS:
                raise ValueError(f"Unknown GenEval2 skill {skill!r} at row {index}")
            skill_scores[skill].append(score)
        atomicity = int(record["atom_count"])
        if atomicity not in ATOMICITIES:
            raise ValueError(f"Unexpected GenEval2 atomicity {atomicity} at row {index}")
        atomicity_scores[atomicity].append(prompt_gm[-1])

    missing_skills = [skill for skill in SKILLS if not skill_scores[skill]]
    missing_atomicities = [value for value in ATOMICITIES if not atomicity_scores[value]]
    if not allow_partial and (missing_skills or missing_atomicities):
        raise ValueError(
            f"Incomplete GenEval2 coverage: missing_skills={missing_skills}, "
            f"missing_atomicities={missing_atomicities}"
        )

    if not allow_partial:
        if len(benchmark_rows) != OFFICIAL_PROMPT_COUNT:
            raise ValueError(
                f"Official GenEval2 requires {OFFICIAL_PROMPT_COUNT} prompts, "
                f"found {len(benchmark_rows)}."
            )
        if question_count != OFFICIAL_QUESTION_COUNT:
            raise ValueError(
                f"Official GenEval2 requires {OFFICIAL_QUESTION_COUNT} questions, "
                f"found {question_count}."
            )

    return {
        "official": not allow_partial,
        "soft_tifa_am": 100.0 * _mean(prompt_am),
        "soft_tifa_gm": 100.0 * _mean(prompt_gm),
        "skills": {
            skill: 100.0 * _mean(skill_scores[skill])
            for skill in SKILLS
            if skill_scores[skill]
        },
        "atomicity": {
            str(value): 100.0 * _mean(atomicity_scores[value])
            for value in ATOMICITIES
            if atomicity_scores[value]
        },
        "num_prompts": len(benchmark_rows),
        "num_images": len(score_lists),
        "num_questions": question_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_data", required=True)
    parser.add_argument("--score_data", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()

    with Path(args.benchmark_data).open("r", encoding="utf-8") as handle:
        benchmark_rows = [json.loads(line) for line in handle if line.strip()]
    score_lists = json.loads(Path(args.score_data).read_text(encoding="utf-8"))
    summary = summarize(benchmark_rows, score_lists, allow_partial=args.allow_partial)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
