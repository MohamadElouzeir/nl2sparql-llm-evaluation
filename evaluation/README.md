# Evaluation Pipeline

This folder defines the fixed, fair protocol for comparing generated SPARQL queries against gold SPARQL.

## 1) Prediction Schema (JSONL)

Each line in `outputs/predictions.jsonl` should contain:

- `question_id` (string or int, matching `uid` in gold file)
- `question`
- `gold_sparql`
- `pred_sparql`
- `method` (`zero_shot`, `retrieval_baseline`, `rag`, `agentic`, ...)
- `model`
- `run_id`
- `confidence`
- `selected_qids` (array)
- `selected_predicates` (array)
- `parse_ok` (bool)
- `generation_latency_ms` (int)
- `exec_ok` (bool)
- `timeout` (bool)
- `latency_ms` (int or null)
- `exec_error` (string or null)
- `pred_answer_canonical` (object or null)

`pipelines/run_batch_generation.py` writes this format.

## 2) Run Generation (Batch, Non-interactive)

Zero-shot:

```bash
python pipelines/run_zero_shot_batch.py \
  --model gpt-5 \
  --input data/LC-QuAD2.0-master/extra/wikidata_with_answers_curated_50.json \
  --output outputs/predictions_zero_shot.jsonl \
  --resume \
  --gen-retries 5 \
  --gen-backoff-initial 1.0 \
  --gen-backoff-max 30 \
  --gen-jitter 0.25 \
  --gen-errors-path outputs/predictions_zero_shot_generation_errors.jsonl \
  --execute
```

Set `OPENAI_API_KEY` in environment before running.

Safety notes:

- Use `--resume` for interrupted runs.
- Generation retries use exponential backoff + jitter.
- Failed generations are logged to `--gen-errors-path` and skipped for later retry.

## 3) Evaluate

```bash
python evaluation/evaluate_predictions.py \
  --gold data/LC-QuAD2.0-master/extra/wikidata_with_answers_curated_50.json \
  --predictions outputs/predictions_zero_shot.jsonl \
  --out-report outputs/eval_report.json \
  --out-items outputs/eval_items.jsonl
```

If predictions were generated without execution metadata:

```bash
python evaluation/evaluate_predictions.py \
  --gold data/LC-QuAD2.0-master/extra/wikidata_with_answers_curated_50.json \
  --predictions outputs/predictions_zero_shot.jsonl \
  --execute-missing
```

## 4) Implemented Metrics

Primary:

- Execution Accuracy
- Answer-set Precision / Recall / F1

Secondary:

- Executable Query Rate
- Syntax Validity Rate
- Triple-pattern Precision / Recall / F1
- Clause Accuracy (`FILTER`, `OPTIONAL`, `UNION`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`, aggregates)
- Entity Precision / Recall / F1 (QIDs)
- Predicate Precision / Recall / F1 (PIDs)
- Latency (mean, median, p95) and Timeout Rate

Breakdowns:

- Overall
- By `method`
- By question type (`subgraph` if present, else heuristic)
