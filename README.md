# IgakuQA119: LLM Evaluation on the 119th Japanese Medical Licensing Examination

IgakuQA119 evaluates large language models on the 119th Japanese Medical Licensing Examination (JMLE, held in February 2025). The repository contains all 400 questions with images, a solver script that supports cloud APIs (OpenAI, Anthropic, Gemini, OpenRouter, PLaMo) and local models via Ollama, and a grading script that updates the leaderboard below. This project was inspired by [nmle-rta](https://github.com/iKora128/nmle-rta).

## Leaderboard

<!-- LEADERBOARD_START -->

| Rank | Entry | Overall Score | Overall Acc. | No-Img Score | No-Img Acc. |
|------|------|--------------|-------------|-------------|------------|
| 1 | Gemini-2.5-Pro | 485/500 (97.00%) | 389/400 (97.25%) | 372/383 (97.13%) | 290/297 (97.64%) |
| 2 | OpenAI-o3 | 482/500 (96.40%) | 384/400 (96.00%) | 370/383 (96.61%) | 286/297 (96.30%) |
| 3 | Gemini-2.5-Flash | 478/500 (95.60%) | 382/400 (95.50%) | 371/383 (96.87%) | 287/297 (96.63%) |
| 4 | Claude-Sonnet-4 | 471/500 (94.20%) | 375/400 (93.75%) | 363/383 (94.78%) | 281/297 (94.61%) |
| 5 | Qwen3-235B-A22B | 462/500 (92.40%) | 366/400 (91.50%) | 356/383 (92.95%) | 274/297 (92.26%) |
| 6 | DeepSeek-R1-0528 | 461/500 (92.20%) | 367/400 (91.75%) | 364/383 (95.04%) | 282/297 (94.95%) |
| 7 | DeepSeek-R1 | 448/500 (89.60%) | 356/400 (89.00%) | 350/383 (91.38%) | 270/297 (90.91%) |
| 8 | Llama4-Maverick | 440/500 (88.00%) | 350/400 (87.50%) | 336/383 (87.73%) | 260/297 (87.54%) |
| 9 | Gemini-2.0-Flash | 436/500 (87.20%) | 352/400 (88.00%) | 333/383 (86.95%) | 263/297 (88.55%) |
| 10 | QwQ-32B | 430/500 (86.00%) | 334/400 (83.50%) | 344/383 (89.82%) | 260/297 (87.54%) |
| 11 | Qwen3-32B | 415/500 (83.00%) | 329/400 (82.25%) | 334/383 (87.21%) | 256/297 (86.20%) |
| 12 | Qwen3-30B-A3B | 412/500 (82.40%) | 328/400 (82.00%) | 323/383 (84.33%) | 251/297 (84.51%) |
| 13 | Qwen2.5-VL-72B | 403/500 (80.60%) | 325/400 (81.25%) | 309/383 (80.68%) | 245/297 (82.49%) |
| 14 | DeepSeek-V3-0324 | 399/500 (79.80%) | 311/400 (77.75%) | 312/383 (81.46%) | 236/297 (79.46%) |
| 15 | Qwen2.5-72B | 398/500 (79.60%) | 314/400 (78.50%) | 311/383 (81.20%) | 241/297 (81.14%) |
| 16 | Cogito-32B-Think | 392/500 (78.40%) | 310/400 (77.50%) | 305/383 (79.63%) | 237/297 (79.80%) |
| 17 | Llama4-Scout | 392/500 (78.40%) | 314/400 (78.50%) | 303/383 (79.11%) | 237/297 (79.80%) |
| 18 | CA-DSR1-DQ32B-JP-SFT | 374/500 (74.80%) | 294/400 (73.50%) | 290/383 (75.72%) | 222/297 (74.75%) |
| 19 | CA-DSR1-DQ32B-JP | 364/500 (72.80%) | 282/400 (70.50%) | 280/383 (73.11%) | 212/297 (71.38%) |
| 20 | CA-DSR1-DQ32B-JP-CPT | 356/500 (71.20%) | 278/400 (69.50%) | 277/383 (72.32%) | 213/297 (71.72%) |
| 21 | Cogito-32B-No-Think | 346/500 (69.20%) | 278/400 (69.50%) | 271/383 (70.76%) | 211/297 (71.04%) |
| 22 | GPT-4o-mini | 345/500 (69.00%) | 279/400 (69.75%) | 269/383 (70.23%) | 215/297 (72.39%) |
| 23 | Preferred-MedLLM-Qwen-72B | 332/500 (66.40%) | 272/400 (68.00%) | 261/383 (68.15%) | 209/297 (70.37%) |
| 24 | MedGemma-27B-Q6_K | 324/500 (64.80%) | 250/400 (62.50%) | 254/383 (66.32%) | 194/297 (65.32%) |
| 25 | Gemma-3-27B | 320/500 (64.00%) | 252/400 (63.00%) | 252/383 (65.80%) | 196/297 (65.99%) |
| 26 | PLaMo-2.0-Prime | 286/500 (57.20%) | 228/400 (57.00%) | 229/383 (59.79%) | 175/297 (58.92%) |
| 27 | PLaMo-1.0-Prime | 211/500 (42.20%) | 175/400 (43.75%) | 156/383 (40.73%) | 126/297 (42.42%) |

<!-- LEADERBOARD_END -->

Scoring follows the official exam: 500 points in total. General questions (blocks A, C, D, F) are worth 1 point each. Required questions (blocks B, E) are worth 3 points for questions 26 to 50 and 1 point otherwise. The No-Img columns count only the 297 questions without images, which is useful for comparing text-only models.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env  # set API keys for the providers you use
```

## Usage

### 1. Solve

`solve.py` runs one question block (A to F) and saves answers to `answers/<exp>.json`.

```bash
# One block
uv run solve.py questions/119A_json.json --model_name gemini-2.5-pro --exp 119A_gemini-2_5-pro

# All blocks
for block in A B C D E F; do
  uv run solve.py questions/119${block}_json.json \
    --model_name gemini-2.5-pro --exp 119${block}_gemini-2_5-pro
done
```

Model names are resolved as follows:

* `gemini-*`: Gemini API (for example `gemini-2.5-pro`)
* `openrouter-<model_id>`: OpenRouter (for example `openrouter-qwen/qwen3-235b-a22b`)
* `ollama-<model_name>` or `hf.co/<user>/<repo>`: local models via Ollama
* `plamo-*`: PLaMo API
* Other keys defined in `solve.py` (for example `gpt-4o`, `o1`, `claude-3.5-sonnet`)

Use `--questions 119A1 119A2` to solve specific questions only (for example to retry failed ones), and `--supports_vision true/false` to override whether images are sent to the model.

### 2. Grade

`grade.py` grades the answer files, prints a summary, and updates `leaderboard.json` and the leaderboard table in this README.

```bash
uv run grade.py -j answers/119*_gemini-2_5-pro.json -e "Gemini-2.5-Pro"
```

You can try it with the bundled demo answers:

```bash
uv run grade.py -j answers/demo/119*_qwen2_5-72b.json -e "Qwen2.5-72B"
```

## Dataset

The question components (text, choices, images) were processed from official exam PDFs using OCR by the author of the original [nmle-rta](https://github.com/iKora128/nmle-rta) repository. Permission for use and publication was obtained.

The grading logic (correct answers, excluded questions handling) was developed based on official MHLW information: [第１１９回医師国家試験の合格発表について](https://www.mhlw.go.jp/general/sikaku/successlist/2025/siken01/about.html).

## License

The source code is licensed under the MIT License. See [LICENSE](LICENSE).

The data in `questions/` and `images/` is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), based on the 119th Japanese Medical Licensing Examination published by the Ministry of Health, Labour and Welfare of Japan. The dataset was curated by [Daichi Nagashima](https://github.com/iKora128) ([nmle-rta](https://github.com/iKora128/nmle-rta)) and further processed by [Naoto Iwase](https://naoto-iwase.github.io/) (this repository).
