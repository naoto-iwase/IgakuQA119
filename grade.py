"""Grade model answers, print a summary, and update the leaderboard and README."""
import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

GENERAL_BLOCKS = ["A", "C", "D", "F"]  # general questions: 1 point each
REQUIRED_BLOCKS = ["B", "E"]           # required questions: 3 points for Q26-50, 1 point otherwise
LEADERBOARD_START = "<!-- LEADERBOARD_START -->"
LEADERBOARD_END = "<!-- LEADERBOARD_END -->"


def normalize_answer(answer: Any) -> str:
    """Normalize multiple-choice notation (e.g. "a, b", "[a,b]") for comparison."""
    s = str(answer).strip().lower() if answer is not None else ""
    for ch in "[], ":
        s = s.replace(ch, "")
    return "".join(sorted(s))


def is_correct_answer(question_number: str, model_answer: Any, correct_answer: str) -> bool:
    normalized = normalize_answer(model_answer)
    if not normalized:
        return False
    if question_number == "119E28":  # officially announced that both a and c are accepted
        return normalized in ("a", "c")
    return normalized == normalize_answer(correct_answer)


def question_points(question_number: str) -> int:
    block = question_number[3]
    if block in REQUIRED_BLOCKS:
        num = int("".join(filter(str.isdigit, question_number[4:])) or 0)
        return 3 if 26 <= num <= 50 else 1
    return 1


def grade_file(json_path: str, correct_answers: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Grade one answer JSON and return (graded rows, skipped question numbers)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for question in data["results"]:
        number = str(question["question_number"])
        answers = question.get("answers", [])
        if not answers or "answer" not in answers[0]:
            skipped.append(number)
            continue
        answer_data = answers[0]
        correct = correct_answers.get(number)
        rows.append({
            "question_number": number,
            "model": answer_data.get("model", "UnknownModel"),
            "model_answer": answer_data["answer"],
            "correct_answer": correct,
            "is_correct": is_correct_answer(number, answer_data["answer"], correct) if correct is not None else None,
            "confidence": answer_data.get("confidence"),
            "has_image": question.get("has_image", False),
        })
    return rows, skipped


def category_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"correct": 0, "total": 0, "accuracy": 0.0, "score": 0.0, "possible_score": 0.0, "score_rate": 0.0}
    correct = int(df["is_correct"].sum())
    total = len(df)
    score = float(df["score"].sum())
    possible = float(df["possible"].sum())
    return {
        "correct": correct, "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "score": score, "possible_score": possible,
        "score_rate": score / possible if possible > 0 else 0.0,
    }


def block_stats(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    if df.empty:
        return {}, {}
    agg = df.groupby("block").agg(
        correct=("is_correct", "sum"), total=("is_correct", "size"),
        score=("score", "sum"), possible=("possible", "sum"),
    ).astype(float)
    agg["accuracy"] = (agg["correct"] / agg["total"]).fillna(0.0)
    agg["score_rate"] = (agg["score"] / agg["possible"]).fillna(0.0)
    return agg["accuracy"].to_dict(), agg["score_rate"].to_dict()


def consolidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall, general/required, no-image, and per-block statistics."""
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["correct_answer"]).copy()
    df["is_correct"] = df["is_correct"].fillna(False).astype(bool)
    df["has_image"] = df["has_image"].fillna(False).astype(bool)
    df["block"] = df["question_number"].str[3]
    df["possible"] = df["question_number"].map(question_points)
    df["score"] = df["possible"].where(df["is_correct"], 0)

    no_image = df[~df["has_image"]]
    stats = {
        "total": category_stats(df),
        "general": category_stats(df[df["block"].isin(GENERAL_BLOCKS)]),
        "required": category_stats(df[df["block"].isin(REQUIRED_BLOCKS)]),
        "no_image": {
            **category_stats(no_image),
            "general": category_stats(no_image[no_image["block"].isin(GENERAL_BLOCKS)]),
            "required": category_stats(no_image[no_image["block"].isin(REQUIRED_BLOCKS)]),
        },
    }
    stats["block_accuracies"], stats["block_score_rates"] = block_stats(df)
    stats["no_image"]["block_accuracies"], stats["no_image"]["block_score_rates"] = block_stats(no_image)
    return stats


def format_rate(rate: float) -> str:
    return f"{rate:.2%}"


def format_score(score: float, possible: float) -> str:
    def fmt(x: float) -> str:
        return f"{int(x)}" if x == int(x) else f"{x:.1f}"
    return f"{fmt(score)} / {fmt(possible)}"


def print_summary(stats: dict[str, Any], entry_name: str) -> None:
    print(f"\n===== Results for entry: {entry_name} =====\n")
    sections = {
        "Overall": stats["total"],
        "General (A,C,D,F)": stats["general"],
        "Required (B,E)": stats["required"],
        "No image - Overall": stats["no_image"],
        "No image - General (A,C,D,F)": stats["no_image"]["general"],
        "No image - Required (B,E)": stats["no_image"]["required"],
    }
    for title, data in sections.items():
        print(f"--- {title} ---")
        print(f"  Correct/Total: {data['correct']} / {data['total']}")
        print(f"  Accuracy: {format_rate(data['accuracy'])}")
        print(f"  Score: {format_score(data['score'], data['possible_score'])}")
        print(f"  Score rate: {format_rate(data['score_rate'])}")
        print()
    for title, acc, sr in (("--- Per-block results (overall) ---", stats["block_accuracies"], stats["block_score_rates"]),
                           ("--- Per-block results (no image) ---", stats["no_image"]["block_accuracies"],
                            stats["no_image"]["block_score_rates"])):
        print(title)
        for block in sorted(acc):
            print(f"  Block {block}: accuracy {format_rate(acc[block])}, score rate {format_rate(sr[block])}")
        print()


def update_leaderboard(leaderboard_path: str, entry_name: str, stats: dict[str, Any]) -> dict[str, Any]:
    path = Path(leaderboard_path)
    leaderboard: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    overall, no_image = stats["total"], stats["no_image"]
    leaderboard[entry_name] = {
        "overall_score": overall["score"],
        "overall_possible_score": overall["possible_score"],
        "overall_score_rate": overall["score_rate"],
        "overall_correct": overall["correct"],
        "overall_total": overall["total"],
        "overall_accuracy": overall["accuracy"],
        "no_image_score": no_image["score"],
        "no_image_possible_score": no_image["possible_score"],
        "no_image_score_rate": no_image["score_rate"],
        "no_image_correct": no_image["correct"],
        "no_image_total": no_image["total"],
        "no_image_accuracy": no_image["accuracy"],
    }

    path.write_text(json.dumps(leaderboard, indent=4, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    print(f"Saved leaderboard data to {leaderboard_path}.")
    return leaderboard


def leaderboard_markdown(leaderboard: dict[str, Any]) -> str:
    def cell(value: float, total: float, rate: float) -> str:
        def fmt(x: float) -> str:
            return f"{int(x)}" if x == int(x) else f"{x:.1f}"
        if total == 0:
            return "N/A"
        return f"{fmt(value)}/{fmt(total)} ({rate:.2%})"

    headers = ["Rank", "Entry", "Overall Score", "Overall Acc.", "No-Img Score", "No-Img Acc."]
    lines = ["| " + " | ".join(headers) + " |",
             "|-" + "-|".join("-" * len(h) for h in headers) + "-|"]
    entries = sorted(leaderboard.items(), key=lambda item: item[1]["overall_score_rate"], reverse=True)
    for rank, (name, s) in enumerate(entries, 1):
        row = [str(rank), str(name).replace("|", "\\|"),
               cell(s["overall_score"], s["overall_possible_score"], s["overall_score_rate"]),
               cell(s["overall_correct"], s["overall_total"], s["overall_accuracy"]),
               cell(s["no_image_score"], s["no_image_possible_score"], s["no_image_score_rate"]),
               cell(s["no_image_correct"], s["no_image_total"], s["no_image_accuracy"])]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def update_readme(readme_path: str, markdown_table: str) -> None:
    path = Path(readme_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if LEADERBOARD_START in line)
    end = next(i for i, line in enumerate(lines) if LEADERBOARD_END in line)
    new_lines = lines[:start + 1] + ["", markdown_table, ""] + lines[end:]
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"Updated the leaderboard in {readme_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade model answers and update the leaderboard",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--json_paths", "-j", nargs="+", required=True,
                        help="paths to model answer JSONs (e.g. answers/119A_my-model.json ...)")
    parser.add_argument("--entry_name", "-e", required=True,
                        help="entry name shown on the leaderboard")
    parser.add_argument("--answers_path", "-a", default="questions/correct_answers.csv",
                        help="path to the correct answers CSV")
    parser.add_argument("--leaderboard", default="leaderboard.json",
                        help="path to the leaderboard JSON")
    parser.add_argument("--readme", default="README.md",
                        help="path to the README containing the leaderboard table")
    args = parser.parse_args()

    correct_df = pd.read_csv(args.answers_path)
    correct_answers = dict(zip(correct_df["問題番号"].astype(str), correct_df["解答"].astype(str)))

    all_rows: list[dict[str, Any]] = []
    all_skipped: list[str] = []
    for json_path in args.json_paths:
        rows, skipped = grade_file(json_path, correct_answers)
        all_rows.extend(rows)
        all_skipped.extend(skipped)

    if all_skipped:
        skipped_list = sorted(set(all_skipped))
        print(f"Warning: {len(skipped_list)} questions have no answer: {skipped_list}")
        print("         You can re-run them with the --questions option of solve.py.")
    if not all_rows:
        print("No valid graded results.")
        return

    stats = consolidate(all_rows)
    print_summary(stats, args.entry_name)
    leaderboard = update_leaderboard(args.leaderboard, args.entry_name, stats)
    update_readme(args.readme, leaderboard_markdown(leaderboard))


if __name__ == "__main__":
    main()
