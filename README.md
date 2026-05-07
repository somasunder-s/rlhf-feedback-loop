# rlhf-feedback-loop

A **human-in-the-loop annotation tool** for capturing structured feedback on LLM outputs — specifically, comparing a fine-tuned local GPT-2 against GPT-4o-mini on a resume-section entity-extraction task. Built with Taipy as the GUI layer.

The captured feedback is the **first half of an RLHF / preference-learning pipeline**: human ratings collected here become training signal for downstream reward modeling or DPO.

## What it does

```
   CSV of resume snippets
            │
            ▼
   for each row:
        ├─► fine-tuned local GPT-2  → JSON: {company, title, dates, ...}
        └─► GPT-4o-mini (OpenAI API) → JSON: {company, title, dates, ...}
                       │
                       ▼
   Taipy UI presents both side-by-side
                       │
        annotator marks each field: correct / incorrect, optional text correction
                       │
                       ▼
   sentence-transformer similarity score (paraphrase-MiniLM-L6-v2)
   between model outputs and human-corrected ground truth
                       │
                       ▼
   feedback row written to CSV — keyed by file_name + actual_row_num
```

## Why

- **Two-model comparison in one pass** — every example is scored against both a small fine-tuned model and a large API model. The captured deltas tell you where the local model is competitive vs. where it's losing.
- **Per-field feedback, not per-example** — extraction tasks have multi-field outputs. A binary "good/bad" loses information; this captures correctness *per field* (company / title / dates / description).
- **Similarity scoring beyond exact match** — sentence-transformer embeddings catch paraphrases and minor wording differences that exact string match would mark as wrong.
- **Output is RLHF-ready** — feedback CSV maps source → both model responses → human-corrected version → field-level scores. Direct input to a reward model or DPO setup.

## Tech stack

- **GUI:** Taipy (`taipy.gui`) — Python-native reactive UI, no JS
- **LLMs:** OpenAI Python SDK (GPT-4o-mini), Hugging Face Transformers (local fine-tuned GPT-2)
- **Embeddings:** `sentence-transformers` — `paraphrase-MiniLM-L6-v2`
- **Data:** pandas (CSV in, CSV out)
- **PyTorch** for the local model

## Project layout

```
src/
├── app.py            Taipy UI: annotation workflow, save handlers
├── model_caller.py   ModelCaller: calls both models, returns paired responses
├── models.py         Gpt4oMini wrapper + local GPT-2 wrapper
├── helpers.py        CSV/JSON I/O, feedback serializer
└── config.py         model name, paths, instruction template
data/
└── prompt.txt        instruction templates
requirements.txt
```

## Running

```bash
export OPENAI_API_KEY=sk-...                          # required
# Place fine-tuned local model under data/models/book_metadata_model/

pip install -r requirements.txt
cd src && python app.py
# → Taipy launches a browser window
```

Workflow:
1. Load a CSV with columns `file_name`, `split_on_tokens`.
2. UI shows source text + both model outputs side-by-side.
3. Mark each extracted field correct / incorrect, type corrections inline.
4. Click "Save & Next". Feedback is appended to the output CSV.

## Notes

- The instruction template is hardcoded for **book-section extraction** today. Switching to education / project sections is a config edit (`INSTRUCTION` + `SAMPLE_OUTPUT` in `config.py`).
- Local model weights are not in the repo. Drop your own fine-tuned GPT-2 into `data/models/...`.
- This tool is the **annotation step**. Reward-model training and policy fine-tuning are downstream.
