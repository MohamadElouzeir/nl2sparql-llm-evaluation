"""Interactive Natural Language -> SPARQL generator.

Run:
    python3 main.py

What this version does:
- Prompts the user for one natural language question with input().
- Loads enriched entities from `entities_rich.txt` (JSONL: one JSON object per line).
- Loads enriched predicates from `predicates_with_frequency_rich.txt` (JSON array).
- Retrieves the top-15 candidate entities and predicates locally.
- Sends the question + retrieved context to OpenAI.
- Saves the full structured result to a JSON file in the output directory.
- Prints ONLY the SPARQL query and confidence in the terminal.

Everything that may change later is defined as a constant near the top.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: openai. Install it with: pip install openai") from exc


#constants
OPENAI_API_KEY = "SECRET"
ENTITIES_FILE = "entities_rich.txt"
PREDICATES_FILE = "predicates_with_frequency_rich.txt"
OUTPUT_DIR = "outputs"
OUTPUT_FILENAME = "generated_sparql_result.json"
TOP_K = 15

SYSTEM_PROMPT_TEMPLATE = """You are an expert Wikidata SPARQL generator.

Rules:
- Produce a valid Wikidata SPARQL query.
- Use the candidate entities and predicates provided in context.
- Prefer the most specific relevant entity IDs and predicate IDs.
- Return a single SPARQL query in the `sparql_query` field.
- Do not add explanations.
- If multiple query forms are possible, choose the simplest correct one.
- Use Wikidata prefixes when appropriate.
- The query should be directly runnable against the Wikidata Query Service.

You will receive:
- a natural language question
- a shortlist of candidate entities
- a shortlist of candidate predicates

Your output must follow the provided JSON schema exactly.
"""

USER_PROMPT_TEMPLATE = """Question:
{question}

Candidate entities (top {top_k}):
{entity_context}

Candidate predicates (top {top_k}):
{predicate_context}

Task:
Generate the best possible Wikidata SPARQL query for the question.
"""


#data structures
@dataclass
class KnowledgeItem:
    id: str
    label: str
    description: str = ""
    aliases: List[str] = None
    frequency: Optional[int] = None

    def __post_init__(self) -> None:
        if self.aliases is None:
            self.aliases = []

    def search_text(self) -> str:
        parts = [self.id, self.label, self.description, " ".join(self.aliases or [])]
        if self.frequency is not None:
            parts.append(str(self.frequency))
        return normalize_text(" ".join(p for p in parts if p))


@dataclass
class RetrievedItem:
    id: str
    label: str
    description: str
    aliases: List[str]
    score: float
    frequency: Optional[int] = None


@dataclass
class GenerationResult:
    question: str
    sparql_query: str
    confidence: float
    selected_qids: List[str]
    selected_predicates: List[str]
    retrieved_entities: List[Dict[str, Any]]
    retrieved_predicates: List[Dict[str, Any]]
    raw_model_output: Dict[str, Any]


# helps with retrieval
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "there", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "with",
    "would", "please", "tell", "me",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(normalize_text(text))
    return [t for t in tokens if t not in STOPWORDS]


def jaccard_score(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def phrase_bonus(query_norm: str, item: KnowledgeItem) -> float:
    bonus = 0.0
    label_norm = normalize_text(item.label)
    if label_norm and (label_norm in query_norm or query_norm in label_norm):
        bonus += 0.9
    for alias in item.aliases or []:
        alias_norm = normalize_text(alias)
        if alias_norm and (alias_norm in query_norm or query_norm in alias_norm):
            bonus += 0.55
    if item.id.lower() in query_norm:
        bonus += 1.2
    return bonus


def item_score(query: str, item: KnowledgeItem) -> float:
    query_norm = normalize_text(query)
    query_tokens = tokenize(query)
    item_tokens = tokenize(item.search_text())

    overlap = jaccard_score(query_tokens, item_tokens)
    ratio = sequence_ratio(query_norm, normalize_text(item.search_text()))
    bonus = phrase_bonus(query_norm, item)

    score = (2.8 * overlap) + (1.5 * ratio) + bonus
    if item.frequency is not None:
        score += min(math.log1p(max(item.frequency, 0)) / 10.0, 0.45)
    return score


def rank_items(query: str, items: List[KnowledgeItem], top_k: int = TOP_K) -> List[RetrievedItem]:
    scored = [
        RetrievedItem(
            id=item.id,
            label=item.label,
            description=item.description,
            aliases=item.aliases or [],
            score=item_score(query, item),
            frequency=item.frequency,
        )
        for item in items
    ]
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]

#loading files
def load_entities_file(path: Path) -> List[KnowledgeItem]:
    items: List[KnowledgeItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                KnowledgeItem(
                    id=obj["id"],
                    label=obj.get("label", ""),
                    description=obj.get("description", ""),
                    aliases=obj.get("aliases", []) or [],
                )
            )
    return items


def load_predicates_file(path: Path) -> List[KnowledgeItem]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    parsed = json.loads(raw)
    items: List[KnowledgeItem] = []
    for obj in parsed:
        items.append(
            KnowledgeItem(
                id=obj["id"],
                label=obj.get("label", ""),
                description=obj.get("description", ""),
                aliases=obj.get("aliases", []) or [],
                frequency=obj.get("frequency"),
            )
        )
    return items



# Prompt formatting and calling API
def format_candidate_block(items: List[RetrievedItem]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(items, start=1):
        alias_text = ", ".join(item.aliases[:6]) if item.aliases else ""
        desc = item.description or ""
        extra = []
        if desc:
            extra.append(f"desc={desc}")
        if alias_text:
            extra.append(f"aliases={alias_text}")
        if item.frequency is not None:
            extra.append(f"frequency={item.frequency}")
        extra_str = " | ".join(extra)
        if extra_str:
            lines.append(f"{idx}. {item.id} :: {item.label} :: {extra_str}")
        else:
            lines.append(f"{idx}. {item.id} :: {item.label}")
    return "\n".join(lines)


def build_json_schema() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "sparql_generation_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "sparql_query": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "selected_qids": {"type": "array", "items": {"type": "string"}},
                "selected_predicates": {"type": "array", "items": {"type": "string"}},
                "reasoning_summary": {"type": "string"},
            },
            "required": [
                "sparql_query",
                "confidence",
                "selected_qids",
                "selected_predicates",
                "reasoning_summary",
            ],
        },
    }


def extract_response_text(response: Any) -> str:
    if hasattr(response, "output_text") and isinstance(response.output_text, str):
        return response.output_text

    if hasattr(response, "output"):
        chunks: List[str] = []
        for item in response.output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        if chunks:
            return "".join(chunks)

    if isinstance(response, dict):
        if "output_text" in response and isinstance(response["output_text"], str):
            return response["output_text"]
    raise ValueError("Could not extract text from the OpenAI response.")


def generate_sparql(
    client: OpenAI,
    question: str,
    retrieved_entities: List[RetrievedItem],
    retrieved_predicates: List[RetrievedItem],
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    entity_context = format_candidate_block(retrieved_entities)
    predicate_context = format_candidate_block(retrieved_predicates)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        entity_context=entity_context,
        predicate_context=predicate_context,
        top_k=top_k,
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_PROMPT_TEMPLATE,
        input=user_prompt,
        text={"format": build_json_schema()},
    )

    text = extract_response_text(response)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {text}") from exc


###
def save_result(result: GenerationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILENAME
    out_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def print_terminal_summary(result: GenerationResult) -> None:
    print(result.sparql_query)
    print(f"Confidence: {result.confidence:.4f}")


#Main
def main() -> None:
    question = input("Enter a natural language question: ").strip()
    if not question:
        raise SystemExit("No question provided.")

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        raise SystemExit("Set OPENAI_API_KEY in the file or as an environment variable.")

    entities_path = Path(ENTITIES_FILE)
    predicates_path = Path(PREDICATES_FILE)
    if not entities_path.exists():
        raise FileNotFoundError(f"Entities file not found: {entities_path}")
    if not predicates_path.exists():
        raise FileNotFoundError(f"Predicates file not found: {predicates_path}")

    entities = load_entities_file(entities_path)
    predicates = load_predicates_file(predicates_path)

    retrieved_entities = rank_items(question, entities, top_k=TOP_K)
    retrieved_predicates = rank_items(question, predicates, top_k=TOP_K)

    client = OpenAI(api_key=api_key)
    model_output = generate_sparql(
        client=client,
        question=question,
        retrieved_entities=retrieved_entities,
        retrieved_predicates=retrieved_predicates,
        top_k=TOP_K,
    )

    result = GenerationResult(
        question=question,
        sparql_query=str(model_output.get("sparql_query", "")).strip(),
        confidence=float(model_output.get("confidence", 0.0)),
        selected_qids=list(model_output.get("selected_qids", [])),
        selected_predicates=list(model_output.get("selected_predicates", [])),
        retrieved_entities=[asdict(x) for x in retrieved_entities],
        retrieved_predicates=[asdict(x) for x in retrieved_predicates],
        raw_model_output=model_output,
    )

    save_result(result, Path(OUTPUT_DIR))
    print_terminal_summary(result)


if __name__ == "__main__":
    main()



# do a log of all the things I learnt at this internship
# jaccard, etc
#set temperature to zero to make randomness low in GPT

# do a baseline that is zero-shot without any other context like the top 15. but the same prompt

#maybe get rid of order by
# check with mohamad