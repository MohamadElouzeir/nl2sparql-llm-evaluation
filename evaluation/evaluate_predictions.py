import argparse
import json
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from rdflib.plugins.sparql.parser import parseQuery

DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_TIMEOUT = 30
DEFAULT_USER_AGENT = "nl2sparql-llm-evaluation/eval (contact: local)"

CLAUSE_PATTERNS = {
    "filter": re.compile(r"\bFILTER\b", re.IGNORECASE),
    "optional": re.compile(r"\bOPTIONAL\b", re.IGNORECASE),
    "union": re.compile(r"\bUNION\b", re.IGNORECASE),
    "group_by": re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE),
    "having": re.compile(r"\bHAVING\b", re.IGNORECASE),
    "order_by": re.compile(r"\bORDER\s+BY\b", re.IGNORECASE),
    "limit": re.compile(r"\bLIMIT\b", re.IGNORECASE),
    "aggregate": re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE),
}

SKIP_TRIPLE_PREFIXES = (
    "FILTER ",
    "OPTIONAL ",
    "UNION ",
    "BIND ",
    "VALUES ",
    "SERVICE ",
    "MINUS ",
    "GRAPH ",
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def detect_question_text(item: Dict[str, Any]) -> str:
    for key in ("paraphrased_question", "question", "NNQT_question"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def to_id(value: Any) -> str:
    return str(value)


def canonicalize_answer(answer: List[Any]) -> Dict[str, Any]:
    if len(answer) == 1 and isinstance(answer[0], bool):
        return {"type": "ask", "value": bool(answer[0])}

    rows: List[List[str]] = []
    flat_values: List[str] = []
    for row in answer:
        if isinstance(row, dict):
            vals = sorted(str(v) for v in row.values())
        else:
            vals = [str(row)]
        rows.append(vals)
        flat_values.extend(vals)

    rows.sort(key=lambda r: json.dumps(r, ensure_ascii=False))
    flat_values = sorted(set(flat_values))
    return {"type": "bindings", "rows": rows, "flat_values": flat_values}


def canonicalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "boolean" in payload:
        return {"type": "ask", "value": bool(payload.get("boolean", False))}

    bindings = payload.get("results", {}).get("bindings", [])
    if not isinstance(bindings, list):
        return {"type": "bindings", "rows": [], "flat_values": []}

    answer_rows: List[Dict[str, str]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        row: Dict[str, str] = {}
        for var_name in sorted(binding.keys()):
            cell = binding.get(var_name)
            if not isinstance(cell, dict):
                continue
            value = cell.get("value")
            if value is None:
                continue
            row[var_name] = str(value)
        if row:
            answer_rows.append(row)
    return canonicalize_answer(answer_rows)


def canonical_to_row_set(answer: Dict[str, Any]) -> Set[Tuple[str, ...]]:
    if answer.get("type") == "ask":
        return {("__ASK__", str(bool(answer.get("value", False))))}
    rows = answer.get("rows", [])
    out: Set[Tuple[str, ...]] = set()
    for row in rows:
        if isinstance(row, list):
            out.add(tuple(str(x) for x in row))
    return out


def safe_prf(gold: Set[Any], pred: Set[Any]) -> Tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    inter = len(gold & pred)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    if p + r == 0:
        return p, r, 0.0
    return p, r, (2.0 * p * r) / (p + r)


def parse_ok(query: str) -> bool:
    if not query.strip():
        return False
    try:
        parseQuery(query)
        return True
    except Exception:
        return False


def normalize_variable(token: str) -> str:
    token = token.strip()
    if token.startswith("?"):
        return "?VAR"
    return token


def extract_where_body(query: str) -> str:
    query = re.sub(r"(?im)^\s*PREFIX\s+[^\n]+\n", "", query)
    match = re.search(
        r"\bWHERE\s*\{(.*)\}\s*(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|OFFSET|$)",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1)
    open_brace = query.find("{")
    close_brace = query.rfind("}")
    if open_brace >= 0 and close_brace > open_brace:
        return query[open_brace + 1 : close_brace]
    return query


def extract_triples(query: str) -> Set[str]:
    where_body = extract_where_body(query)
    candidates = where_body.split(".")
    triples: Set[str] = set()
    for raw in candidates:
        line = re.sub(r"\s+", " ", raw.strip())
        if not line:
            continue
        upper = line.upper()
        if any(upper.startswith(prefix) for prefix in SKIP_TRIPLE_PREFIXES):
            continue
        parts = line.split(" ")
        if len(parts) < 3:
            continue
        subj = normalize_variable(parts[0])
        pred = normalize_variable(parts[1])
        obj = normalize_variable(" ".join(parts[2:]))
        triples.add(f"{subj} {pred} {obj}")
    return triples


def extract_clauses(query: str) -> Dict[str, bool]:
    present: Dict[str, bool] = {}
    for clause_name, pattern in CLAUSE_PATTERNS.items():
        present[clause_name] = bool(pattern.search(query))
    return present


def clause_accuracy(gold_query: str, pred_query: str) -> float:
    gold = extract_clauses(gold_query)
    pred = extract_clauses(pred_query)
    keys = sorted(gold.keys())
    if not keys:
        return 1.0
    hits = sum(1 for key in keys if gold[key] == pred[key])
    return hits / len(keys)


QID_PATTERNS = [
    re.compile(r"\bwd:(Q\d+)\b", re.IGNORECASE),
    re.compile(r"/entity/(Q\d+)\b", re.IGNORECASE),
]

PID_PATTERNS = [
    re.compile(r"\b(?:wdt|p|ps|pq):(P\d+)\b", re.IGNORECASE),
    re.compile(r"/entity/(P\d+)\b", re.IGNORECASE),
]


def extract_qids(query: str) -> Set[str]:
    qids: Set[str] = set()
    for pattern in QID_PATTERNS:
        for match in pattern.findall(query):
            qids.add(match.upper())
    return qids


def extract_pids(query: str) -> Set[str]:
    pids: Set[str] = set()
    for pattern in PID_PATTERNS:
        for match in pattern.findall(query):
            pids.add(match.upper())
    return pids


def run_query(
    endpoint: str,
    query: str,
    timeout: int,
    user_agent: str,
) -> Dict[str, Any]:
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": user_agent,
    }
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
            "latency_ms": latency_ms,
            "exec_error": None,
            "pred_answer_canonical": canonicalize_payload(payload),
        }
    except requests.Timeout:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "exec_ok": False,
            "timeout": True,
            "latency_ms": latency_ms,
            "exec_error": "Timeout",
            "pred_answer_canonical": None,
        }
    except requests.RequestException as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "exec_ok": False,
            "timeout": False,
            "latency_ms": latency_ms,
            "exec_error": f"RequestException: {exc}",
            "pred_answer_canonical": None,
        }
    except ValueError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "exec_ok": False,
            "timeout": False,
            "latency_ms": latency_ms,
            "exec_error": f"JSONDecodeError: {exc}",
            "pred_answer_canonical": None,
        }


def detect_question_type(item: Dict[str, Any], gold_query: str) -> str:
    subgraph = item.get("subgraph")
    if isinstance(subgraph, str) and subgraph.strip():
        return subgraph.strip()

    q = gold_query.upper()
    if re.search(r"\bASK\b", q):
        return "boolean"
    if re.search(r"\bCOUNT\s*\(", q):
        return "count"
    if re.search(r"\bORDER\s+BY\b|\bLIMIT\b", q):
        return "superlative_or_rank"
    if re.search(r"\bFILTER\b", q):
        return "filter"
    return "other"


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows)}

    parse_vals = [1.0 if bool(r.get("parse_ok")) else 0.0 for r in rows]
    exec_vals = [1.0 if bool(r.get("exec_ok")) else 0.0 for r in rows]
    timeout_vals = [1.0 if bool(r.get("timeout")) else 0.0 for r in rows]
    out["syntax_validity_rate"] = mean(parse_vals)
    out["executable_query_rate"] = mean(exec_vals)
    out["timeout_rate"] = mean(timeout_vals)

    latency_values = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    out["latency_ms_mean"] = mean(latency_values)
    if latency_values:
        lat_sorted = sorted(latency_values)
        p95_index = int(0.95 * (len(lat_sorted) - 1))
        out["latency_ms_median"] = lat_sorted[len(lat_sorted) // 2]
        out["latency_ms_p95"] = lat_sorted[p95_index]
    else:
        out["latency_ms_median"] = None
        out["latency_ms_p95"] = None

    for metric in (
        "execution_accuracy",
        "answer_precision",
        "answer_recall",
        "answer_f1",
        "triple_precision",
        "triple_recall",
        "triple_f1",
        "clause_accuracy",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "predicate_precision",
        "predicate_recall",
        "predicate_f1",
    ):
        vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
        out[metric] = mean(vals)
        out[metric + "_coverage"] = len(vals)
    return out


def load_gold(path: str) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Gold file must be a JSON array: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        qid = row.get("uid", idx)
        qid_str = to_id(qid)
        out[qid_str] = row
    return out


def answer_metrics(
    gold_answer: Optional[Dict[str, Any]],
    pred_answer: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if gold_answer is None or pred_answer is None:
        return None, None, None, None
    execution_accuracy = 1.0 if gold_answer == pred_answer else 0.0
    gold_rows = canonical_to_row_set(gold_answer)
    pred_rows = canonical_to_row_set(pred_answer)
    p, r, f1 = safe_prf(gold_rows, pred_rows)
    return execution_accuracy, p, r, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NL2SPARQL predictions against gold data.")
    parser.add_argument(
        "--gold",
        default="data/LC-QuAD2.0-master/extra/wikidata_with_answers_curated_50.json",
        help="Gold JSON containing uid, question, sparql_wikidata, and gold answers.",
    )
    parser.add_argument("--predictions", required=True, help="Predictions JSONL file.")
    parser.add_argument("--out-report", default="outputs/eval_report.json", help="Aggregate report JSON.")
    parser.add_argument("--out-items", default="outputs/eval_items.jsonl", help="Per-item metrics JSONL.")
    parser.add_argument("--execute-missing", action="store_true", help="Execute predicted SPARQL when no answers are present.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    args = parser.parse_args()

    gold_map = load_gold(args.gold)
    predictions = load_jsonl(args.predictions)

    item_rows: List[Dict[str, Any]] = []
    missing_gold = 0
    for pred in predictions:
        pred_id = to_id(pred.get("question_id"))
        gold = gold_map.get(pred_id)
        if gold is None:
            missing_gold += 1
            continue

        gold_query = str(gold.get("sparql_wikidata", ""))
        pred_query = str(pred.get("pred_sparql", "")).strip()
        row: Dict[str, Any] = {
            "question_id": pred_id,
            "question": detect_question_text(gold),
            "method": pred.get("method"),
            "model": pred.get("model"),
            "run_id": pred.get("run_id"),
            "question_type": detect_question_type(gold, gold_query),
            "gold_sparql": gold_query,
            "pred_sparql": pred_query,
        }

        row["parse_ok"] = bool(pred.get("parse_ok")) if "parse_ok" in pred else parse_ok(pred_query)

        row["exec_ok"] = pred.get("exec_ok")
        row["timeout"] = pred.get("timeout", False)
        row["exec_error"] = pred.get("exec_error")
        row["latency_ms"] = pred.get("latency_ms")
        pred_answer = pred.get("pred_answer_canonical")
        if pred_answer is None:
            alt = pred.get("pred_answers_norm")
            if isinstance(alt, list):
                pred_answer = canonicalize_answer(alt)

        if args.execute_missing and pred_answer is None and pred_query:
            exec_result = run_query(
                endpoint=args.endpoint,
                query=pred_query,
                timeout=args.timeout,
                user_agent=args.user_agent,
            )
            row["exec_ok"] = exec_result["exec_ok"]
            row["timeout"] = exec_result["timeout"]
            row["exec_error"] = exec_result["exec_error"]
            row["latency_ms"] = exec_result["latency_ms"]
            pred_answer = exec_result["pred_answer_canonical"]

        if row.get("exec_ok") is None:
            row["exec_ok"] = False

        gold_answer = gold.get("gold_answer_canonical")
        if gold_answer is None:
            answer_raw = gold.get("answer")
            if isinstance(answer_raw, list):
                gold_answer = canonicalize_answer(answer_raw)

        execution_accuracy, ans_p, ans_r, ans_f1 = answer_metrics(gold_answer, pred_answer)
        row["execution_accuracy"] = execution_accuracy
        row["answer_precision"] = ans_p
        row["answer_recall"] = ans_r
        row["answer_f1"] = ans_f1

        gold_triples = extract_triples(gold_query)
        pred_triples = extract_triples(pred_query)
        t_p, t_r, t_f1 = safe_prf(gold_triples, pred_triples)
        row["triple_precision"] = t_p
        row["triple_recall"] = t_r
        row["triple_f1"] = t_f1

        row["clause_accuracy"] = clause_accuracy(gold_query, pred_query)

        gold_entities = extract_qids(gold_query)
        pred_entities = extract_qids(pred_query)
        e_p, e_r, e_f1 = safe_prf(gold_entities, pred_entities)
        row["entity_precision"] = e_p
        row["entity_recall"] = e_r
        row["entity_f1"] = e_f1

        gold_predicates = extract_pids(gold_query)
        pred_predicates = extract_pids(pred_query)
        p_p, p_r, p_f1 = safe_prf(gold_predicates, pred_predicates)
        row["predicate_precision"] = p_p
        row["predicate_recall"] = p_r
        row["predicate_f1"] = p_f1

        item_rows.append(row)

    grouped_by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped_by_qtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        method_name = str(row.get("method", "unknown"))
        qtype = str(row.get("question_type", "unknown"))
        grouped_by_method[method_name].append(row)
        grouped_by_qtype[qtype].append(row)

    report = {
        "meta": {
            "gold_path": args.gold,
            "predictions_path": args.predictions,
            "execute_missing": args.execute_missing,
            "endpoint": args.endpoint if args.execute_missing else None,
            "timeout_seconds": args.timeout if args.execute_missing else None,
            "missing_gold_prediction_rows": missing_gold,
            "scored_rows": len(item_rows),
        },
        "overall": summarize_group(item_rows),
        "by_method": {method: summarize_group(rows) for method, rows in grouped_by_method.items()},
        "by_question_type": {qtype: summarize_group(rows) for qtype, rows in grouped_by_qtype.items()},
    }

    write_json(args.out_report, report)
    write_jsonl(args.out_items, item_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

