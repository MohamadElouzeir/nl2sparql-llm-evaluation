"""Zero-shot Natural Language -> SPARQL generator.

Run:
    python3 main_zero_shot.py

What this version does:
- Prompts the user for one natural language question with input().
- Sends only the question to OpenAI, no QID/predicate/entity context.
- Saves the full structured result to a JSON file in the output directory.
- Prints ONLY the SPARQL query and confidence in the terminal.

Everything that may change later is defined as a constant near the top.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: openai. Install it with: pip install openai") from exc


#constants
OPENAI_API_KEY = "SECRET"
OPENAI_MODEL = "gpt-5"
OUTPUT_DIR = "outputs"
OUTPUT_FILENAME = "generated_sparql_result.json"

SYSTEM_PROMPT_TEMPLATE = """You are an expert Wikidata SPARQL generator.

Rules:
- Produce a valid Wikidata SPARQL query from the user's natural language question.
- Do not ask follow-up questions.
- Do not provide explanations.
- Return the simplest correct query you can.
- Use Wikidata prefixes when appropriate.
- The query should be directly runnable against the Wikidata Query Service.
- If the question is ambiguous, make the most likely reasonable assumption.

Your output must follow the provided JSON schema exactly.
"""

USER_PROMPT_TEMPLATE = """Question:
{question}

Task:
Generate the best possible Wikidata SPARQL query for the question.
"""


# data structure
@dataclass
class GenerationResult:
    question: str
    sparql_query: str
    confidence: float
    selected_qids: List[str]
    selected_predicates: List[str]
    raw_model_output: Dict[str, Any]


#Openai stuff
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


def generate_sparql(client: OpenAI, question: str) -> Dict[str, Any]:
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question)

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


#output info
def save_result(result: GenerationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILENAME
    out_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def print_terminal_summary(result: GenerationResult) -> None:
    print(result.sparql_query)
    print(f"Confidence: {result.confidence:.4f}")



# Main

def main() -> None:
    question = input("Enter a natural language question: ").strip()
    if not question:
        raise SystemExit("No question provided.")

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        raise SystemExit("Set OPENAI_API_KEY in the file or as an environment variable.")

    client = OpenAI(api_key=api_key)
    model_output = generate_sparql(client=client, question=question)

    result = GenerationResult(
        question=question,
        sparql_query=str(model_output.get("sparql_query", "")).strip(),
        confidence=float(model_output.get("confidence", 0.0)),
        selected_qids=list(model_output.get("selected_qids", [])),
        selected_predicates=list(model_output.get("selected_predicates", [])),
        raw_model_output=model_output,
    )

    save_result(result, Path(OUTPUT_DIR))
    print_terminal_summary(result)


if __name__ == "__main__":
    main()
