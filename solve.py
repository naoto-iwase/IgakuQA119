"""Solve JMLE questions with an LLM and save the answers to answers/ as JSON."""
import argparse
import base64
import copy
import glob
import json
import os
import random
import re
import textwrap
import time
import traceback
from datetime import datetime
from typing import Any, Optional

import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# The prompts are written in Japanese on purpose: the exam itself is in Japanese,
# and all leaderboard entries were evaluated with these exact prompts.
SYSTEM_PROMPT = textwrap.dedent("""\
    あなたは医師国家試験問題を解く優秀で論理的なアシスタントです。
    以下のルールを守って、問題文と選択肢（または数値入力の指示）を確認し、回答してください。

    【ルール】
    1. 明示的な指示がない場合は、単一選択肢のみ選ぶ（例: "a", "d"）。
    2. 「2つ選べ」「3つ選べ」などとあれば、その数だけ選択肢をアルファベット順で列挙する（例: "ac", "bd"）。
    3. 選択肢が存在せず数値入力が求められる場合は、指定がない限りそのままの数値を答える（例: answer: 42）。
    4. 画像（has_image=True）は参考情報とし、特別な形式は不要。
    5. 不要な装飾やMarkdown記法は含めず、以下の形式に従って厳密に出力してください：

    answer: [選んだ回答(単数/複数/数値)]
    confidence: [0.0～1.0の確信度]
    explanation: [選択理由や重要な根拠を簡潔に]

    【answerについて注意】
    - 問題は単数選択、複数選択、数値入力のいずれかであり、問題文からその形式を判断する。
    - 「どれか。」で終わる選択問題で数が明記されていない場合は、五者択一を意味するので選択肢を必ず1つだけ選び小文字のアルファベットで回答する。（単数選択）
    - 「2つ選べ」「3つ選べ」などと書いてある場合に限り、指定された数だけの複数選択肢を選び、小文字のアルファベット順（abcde順）に並び替えて列挙する。（複数選択）
    - 選択肢が存在しない場合は、小数や四捨五入など、問題文で特に指示があればそれに従い、選択肢記号ではなく数値を回答する。（数値入力）
    - 問題に関連しない余計な文は書かず、指定のキー(answer, confidence, explanation)を上記の出力に従って厳密に出力する。
""")

MODELS: dict[str, dict[str, Any]] = {
    "o1": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": "o1",
        "client_type": "openai",
        "supports_vision": True,
        "system_role": "system",
        "parameters": {},
    },
    "gpt-4o": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": "gpt-4o",
        "client_type": "openai",
        "supports_vision": True,
        "system_role": "system",
        "parameters": {},
    },
    "o3-mini": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": "o3-mini",
        "client_type": "openai",
        "supports_vision": False,
        "system_role": "system",
        "parameters": {"reasoning_effort": "high"},
    },
    "claude-3.5-sonnet": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "model_name": "claude-3-5-sonnet-20241022",
        "client_type": "anthropic",
        "supports_vision": True,
        "system_role": "system",
        "parameters": {"temperature": 0.2, "max_tokens": 1000},
    },
    "gemma-3": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/",
        "model_name": "gemma-3-27b-it",
        "client_type": "openai",
        "supports_vision": False,
        "system_role": "user",
        "parameters": {},
    },
    "gemini-flexible": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/",
        "model_name": None,
        "client_type": "openai",
        "supports_vision": True,
        "system_role": "system",
        "parameters": {},
    },
    "openrouter-flexible": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
        "model_name": None,
        "client_type": "openai",
        "supports_vision": False,
        "system_role": "system",
        "parameters": {},
        "extra_body": {"enable_thinking": True},
    },
    "ollama-flexible": {
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
        "model_name": None,
        "client_type": "openai",
        "supports_vision": False,
        "system_role": "system",
        "parameters": {},
    },
    "plamo-flexible": {
        "api_key": os.getenv("PLAMO_API_KEY"),
        "base_url": "https://platform.preferredai.jp/api/completion/v1",
        "model_name": None,
        "client_type": "openai",
        "supports_vision": False,
        "system_role": "user",
        "parameters": {},
    },
}

COT_PATTERN = re.compile(r"<(think|thinking|thoughts)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


def resolve_model(model_key: str) -> tuple[dict[str, Any], Any]:
    """Return the config and API client for a model key. Unknown keys are resolved by prefix."""
    if model_key in MODELS:
        config = copy.deepcopy(MODELS[model_key])
    elif model_key.startswith("gemini-"):
        config = copy.deepcopy(MODELS["gemini-flexible"])
        config["model_name"] = model_key
    elif model_key.startswith(("hf.co/", "huggingface.co/")):
        config = copy.deepcopy(MODELS["ollama-flexible"])
        config["model_name"] = model_key
    elif model_key.startswith("ollama-"):
        config = copy.deepcopy(MODELS["ollama-flexible"])
        config["model_name"] = model_key.split("ollama-", 1)[1]
    elif model_key.startswith("openrouter-"):
        config = copy.deepcopy(MODELS["openrouter-flexible"])
        config["model_name"] = model_key.split("openrouter-", 1)[1]
    elif model_key.startswith("plamo-"):
        config = copy.deepcopy(MODELS["plamo-flexible"])
        config["model_name"] = model_key
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    if config["client_type"] == "anthropic":
        client = anthropic.Anthropic(api_key=config["api_key"])
    else:
        client_args: dict[str, Any] = {"api_key": config["api_key"]}
        if config.get("base_url"):
            client_args["base_url"] = config["base_url"]
        client = OpenAI(**client_args)
    return config, client


def parse_response(response: str, cot: Optional[str]) -> tuple[dict[str, Any], bool]:
    """Extract answer / confidence / explanation / cot from a raw model response."""
    result: dict[str, Any] = {"answer": None, "confidence": None, "explanation": None, "cot": cot}
    if not response or not isinstance(response, str):
        return result, False

    if cot is None:
        cot_match = COT_PATTERN.search(response)
        if cot_match:
            result["cot"] = cot_match.group(2).strip()
    cleaned = COT_PATTERN.sub("", response).strip()

    success = True
    expl_lines: list[str] = []
    found_expl = False
    for line in cleaned.lower().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("answer:"):
            result["answer"] = line.removeprefix("answer:").strip()
        elif line.startswith("confidence:"):
            match = re.search(r"(\d\.?\d*)", line.removeprefix("confidence:"))
            if match:
                result["confidence"] = max(0.0, min(1.0, float(match.group(1))))
            else:
                success = False
        elif line.startswith("explanation:"):
            found_expl = True
            first = line.removeprefix("explanation:").strip()
            if first:
                expl_lines.append(first)
        elif found_expl:
            expl_lines.append(line)

    if expl_lines:
        result["explanation"] = "\n".join(expl_lines)
    elif found_expl:
        result["explanation"] = ""
    else:
        result["explanation"] = cleaned

    if not result["answer"]:
        success = False
    return result, success


def question_images(question_number: str) -> list[str]:
    paths: list[str] = []
    for pattern in (f"images/{question_number}.jpg", f"images/{question_number}.png",
                    f"images/{question_number}-*.jpg", f"images/{question_number}-*.png"):
        paths.extend(glob.glob(pattern))
    return sorted(paths)


def build_user_content(question: dict[str, Any], prompt: str, use_vision: bool) -> Any:
    """Build the user message content, either text only or text with images."""
    if not (use_vision and question.get("has_image", False)):
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in question_images(question["number"]):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    if len(content) == 1:
        print(f"Warning: question {question['number']} has has_image=True but no image files were found.")
        return prompt
    return content


def call_model(config: dict[str, Any], client: Any, user_content: Any) -> tuple[str, Optional[str]]:
    """Make one API call and return (response text, chain of thought)."""
    if config["client_type"] == "anthropic":
        response = client.messages.create(
            model=config["model_name"],
            messages=[{"role": "user", "content": user_content}],
            system=SYSTEM_PROMPT,
            **config["parameters"],
        )
        return response.content[0].text, None
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[
            {"role": config["system_role"], "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        extra_body=config.get("extra_body", {}),
        **config["parameters"],
    )
    message = response.choices[0].message
    return message.content, getattr(message, "reasoning", None)


def solve_question(question: dict[str, Any], model_key: str,
                   supports_vision_override: Optional[bool] = None) -> dict[str, Any]:
    """Solve one question. API errors are retried up to 5 times with exponential backoff."""
    config, client = resolve_model(model_key)
    use_vision = config["supports_vision"] if supports_vision_override is None else supports_vision_override

    prompt = (f"問題：{question['question']}\n\n"
              f"選択肢：\n{chr(10).join(question['choices'])}\n\n"
              "回答を指定された形式で出力してください。")
    # The anthropic client path is text only, as in the original evaluation setup.
    user_content = prompt if config["client_type"] == "anthropic" else build_user_content(question, prompt, use_vision)

    max_retries = 5
    last_error: Exception = RuntimeError("unreachable")
    for attempt in range(max_retries):
        try:
            raw_response, cot = call_model(config, client, user_content)
            parsed, success = parse_response(raw_response, cot)

            # If the response is malformed, ask the model itself to reformat it.
            for _ in range(3):
                if success:
                    break
                print("Retrying with a reformatting request.")
                retry_prompt = (f"以下の「整形前の応答」を、「整形方法の指示」にて指定された形式に厳密に整形してください。\n"
                                f"【整形前の応答】\n{raw_response}\n\n【整形方法の指示】\n{SYSTEM_PROMPT}")
                fixed, _ = call_model(config, client, retry_prompt)
                parsed, success = parse_response(fixed, cot)
                if success:
                    raw_response = fixed
            if not success:
                raise RuntimeError("failed to reformat the response into the required format")

            return {
                "model_used": model_key,
                "raw_response": raw_response,
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "explanation": parsed["explanation"],
                "cot": parsed["cot"],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"Error from model {model_key}: {e}. Retrying in {wait:.2f}s ({attempt + 1}/{max_retries})")
                time.sleep(wait)

    print(f"Reached the maximum number of retries for model {model_key}. Last error: {last_error}")
    return {
        "model_used": model_key,
        "error": repr(last_error),
        "traceback": "".join(traceback.format_exception(last_error)),
        "timestamp": datetime.now().isoformat(),
    }


def save_answers(results: list[dict[str, Any]], file_exp: str) -> None:
    """Save the results to answers/{file_exp}.json."""
    os.makedirs("answers", exist_ok=True)
    output_file = os.path.join("answers", f"{os.path.basename(file_exp)}.json")

    formatted = []
    for result in results:
        entry = {
            "question_number": result["question"]["number"],
            "question_text": result["question"]["question"],
            "choices": result["question"]["choices"],
            "has_image": result["question"].get("has_image", False),
            "answers": [],
        }
        for answer in result["answers"]:
            item: dict[str, Any] = {
                "model": answer["model_used"],
                "timestamp": answer["timestamp"],
            }
            if "error" in answer:
                item["error"] = answer["error"]
            else:
                item.update(answer=answer["answer"], confidence=answer["confidence"],
                            explanation=answer["explanation"], cot=answer["cot"])
            entry["answers"].append(item)
        formatted.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"experiment_id": file_exp, "results": formatted}, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve JMLE questions with an LLM")
    parser.add_argument("input_json", help="path to a question JSON (e.g. questions/119A_json.json)")
    parser.add_argument("--model_name", required=True,
                        help="model name (e.g. gpt-4o, gemini-2.5-pro, ollama-llama3, openrouter-<model_id>)")
    parser.add_argument("--questions", nargs="+",
                        help="question numbers to solve. Solves all questions if omitted (e.g. 119A1 119A2)")
    parser.add_argument("--exp", default=None,
                        help="identifier used in the output filename. Defaults to a timestamp (e.g. 119A_my-model)")
    parser.add_argument("--supports_vision", choices=["true", "false"], default=None,
                        help="explicitly override whether images are sent to the model")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if args.questions:
        questions = [q for q in questions if q["number"] in args.questions]
        if not questions:
            print(f"No questions matched: {args.questions}")
            return

    file_exp = args.exp or datetime.now().strftime("%Y%m%d_%H%M%S")
    vision_override = None if args.supports_vision is None else args.supports_vision == "true"

    results: list[dict[str, Any]] = []
    for question in tqdm(questions, desc="Solving questions"):
        answer = solve_question(question, args.model_name, supports_vision_override=vision_override)
        results.append({"question": question, "answers": [answer]})
        save_answers(results, file_exp)  # save after every question in case of interruption

    print(f"Done. Results: answers/{os.path.basename(file_exp)}.json")


if __name__ == "__main__":
    main()
