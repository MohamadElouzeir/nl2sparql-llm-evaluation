import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set

import requests
from rdflib.plugins.sparql.parser import parseQuery

DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_TIMEOUT = 30
DEFAULT_USER_AGENT = "nl2sparql-llm-evaluation/generation (contact: local)"


@dataclass
class BatchConfig:
    input_path: str
    output_path: str
    method: str
    model: str
    run_id: str
    start_index: int = 0
    max_questions: int = -1
    resume: bool = False
    execute: bool = False
    endpoint: str = DEFAULT_ENDPOINT
    timeout: int = DEFAULT_TIMEOUT
    user_agent: str = DEFAULT_USER_AGENT
    generation_retries: int = 5
    generation_backoff_initial: float = 1.0
    generation_backoff_max: float = 30.0
    generation_jitter: float = 0.25
    generation_errors_path: str = ""


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_question(item: Dict[str, Any]) -> str:
    for key in ("paraphrased_question", "question", "NNQT_question"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_ok(query: str) -> bool:
    if not query.strip():
        return False
    try:
        parseQuery(query)
        return True
    except Exception:
        return False


def canonicalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "boolean" in payload:
        return {"type": "ask", "value": bool(payload.get("boolean", False))}

    rows = payload.get("results", {}).get("bindings", [])
    if not isinstance(rows, list):
        return {"type": "bindings", "rows": [], "flat_values": []}

    normalized_rows: List[List[str]] = []
    flat_values: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        values: List[str] = []
        for var_name in sorted(row.keys()):
            cell = row.get(var_name)
            if not isinstance(cell, dict):
                continue
            value = cell.get("value")
            if value is None:
                continue
            value_str = str(value)
            values.append(value_str)
            flat_values.add(value_str)
        if values:
            normalized_rows.append(sorted(values))

    normalized_rows.sort(key=lambda r: json.dumps(r, ensure_ascii=False))
    return {
        "type": "bindings",
        "rows": normalized_rows,
        "flat_values": sorted(flat_values),
    }


def execute_query(endpoint: str, query: str, timeout: int, user_agent: str) -> Dict[str, Any]:
    headers = {"Accept": "application/sparql-results+json", "User-Agent": user_agent}
    params = {"format": "json"}
    data = {"query": query}
    start = time.perf_counter()
    try:
        response = requests.post(endpoint, params=params, data=data, headers=headers, timeout=timeout)
        latency_ms = int((time.perf_counter() - start) * 1000)
        response.raise_for_status()
        payload = response.json()
        return {
            "exec_ok": True,
            "timeout": False,
            "exec_error": None,
            "latency_ms": latency_ms,
            "pred_answer_canonical": canonicalize_payload(payload),
        }
    except requests.Timeout:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "exec_ok": False,
            "timeout": True,
            "exec_error": "Timeout",
            "latency_ms": latency_ms,
            "pred_answer_canonical": None,
        }
    except requests.RequestException as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "exec_ok": False,
            "timeout": False,
            "exec_error": f"RequestException: {exc}",
            "latency_ms": latency_ms,
            "pred_answer_canonical": None,
        }
    except ValueError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "exec_ok": False,
            "timeout": False,
            "exec_error": f"JSONDecodeError: {exc}",
            "latency_ms": latency_ms,
            "pred_answer_canonical": None,
        }


def load_existing_ids(path: str) -> Set[str]:
    existing: Set[str] = set()
    if not os.path.exists(path):
        return existing
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(obj.get("question_id", ""))
            if qid:
                existing.add(qid)
    return existing


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_run_id(explicit: str = "") -> str:
    if explicit:
        return explicit
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"


def run_batch_generation(
    config: BatchConfig,
    generate_fn: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    data = load_json(config.input_path)
    if not isinstance(data, list):
        raise SystemExit("Input must be a JSON array.")

    if config.start_index > 0:
        data = data[config.start_index :]
    if config.max_questions >= 0:
        data = data[: config.max_questions]

    existing_ids: Set[str] = load_existing_ids(config.output_path) if config.resume else set()
    total = len(data)
    written = 0
    skipped = 0
    generation_failures = 0

    for idx, item in enumerate(data, start=1):
        question_id = str(item.get("uid", idx - 1))
        if question_id in existing_ids:
            skipped += 1
            continue

        question = detect_question(item)
        if not question:
            skipped += 1
            continue

        gen_start = time.perf_counter()
        model_output: Dict[str, Any] | None = None
        generation_error: str | None = None
        for attempt in range(config.generation_retries + 1):
            try:
                model_output = generate_fn(question)
                generation_error = None
                break
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
                if attempt >= config.generation_retries:
                    break
                base_wait = min(
                    config.generation_backoff_initial * (2**attempt),
                    config.generation_backoff_max,
                )
                jitter = random.uniform(0.0, max(config.generation_jitter, 0.0))
                time.sleep(base_wait + jitter)

        if model_output is None:
            generation_failures += 1
            if config.generation_errors_path:
                append_jsonl(
                    config.generation_errors_path,
                    {
                        "question_id": question_id,
                        "question": question,
                        "method": config.method,
                        "model": config.model,
                        "run_id": config.run_id,
                        "error": generation_error,
                    },
                )
            print(f"[{idx}/{total}] generation failed question_id={question_id}")
            continue

        generation_latency_ms = int((time.perf_counter() - gen_start) * 1000)
        pred_query = str(model_output.get("sparql_query", "")).strip()

        row: Dict[str, Any] = {
            "question_id": question_id,
            "question": question,
            "gold_sparql": item.get("sparql_wikidata"),
            "pred_sparql": pred_query,
            "method": config.method,
            "model": config.model,
            "run_id": config.run_id,
            "confidence": float(model_output.get("confidence", 0.0)),
            "selected_qids": list(model_output.get("selected_qids", [])),
            "selected_predicates": list(model_output.get("selected_predicates", [])),
            "parse_ok": parse_ok(pred_query),
            "generation_latency_ms": generation_latency_ms,
            "exec_ok": False,
            "timeout": False,
            "latency_ms": None,
            "exec_error": None,
            "pred_answer_canonical": None,
        }

        if config.execute and pred_query:
            row.update(
                execute_query(
                    endpoint=config.endpoint,
                    query=pred_query,
                    timeout=config.timeout,
                    user_agent=config.user_agent,
                )
            )

        append_jsonl(config.output_path, row)
        written += 1
        print(f"[{idx}/{total}] wrote question_id={question_id}")

    summary = {
        "input": config.input_path,
        "output": config.output_path,
        "method": config.method,
        "model": config.model,
        "run_id": config.run_id,
        "total_seen": total,
        "written": written,
        "skipped": skipped,
        "generation_failures": generation_failures,
        "resume": config.resume,
        "execute": config.execute,
    }
    print(json.dumps(summary, indent=2))
    return summary
