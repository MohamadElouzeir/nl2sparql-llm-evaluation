import argparse
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from run_batch_generation import BatchConfig, make_run_id, run_batch_generation

DEFAULT_INPUT = "data/LC-QuAD2.0-master/extra/wikidata_with_answers.json"
DEFAULT_OUTPUT = "outputs/predictions_zero_shot.jsonl"
DEFAULT_MODEL = "gpt-5"
DEFAULT_ERRORS = "outputs/predictions_zero_shot_generation_errors.jsonl"

SYSTEM_PROMPT_ZERO = """You are an expert Wikidata SPARQL generator.

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

USER_PROMPT_ZERO = """Question:
{question}

Task:
Generate the best possible Wikidata SPARQL query for the question.
"""


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
            },
            "required": ["sparql_query"],
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
        text = response.get("output_text")
        if isinstance(text, str):
            return text

    raise ValueError("Could not extract text from model response.")


def make_zero_shot_generator(client: OpenAI, model: str):
    def generate(question: str) -> Dict[str, Any]:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT_ZERO,
            input=USER_PROMPT_ZERO.format(question=question),
            text={"format": build_json_schema()},
        )
        return json.loads(extract_response_text(response))

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run zero-shot batch generation.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--user-agent", default=None)
    parser.add_argument("--gen-retries", type=int, default=5)
    parser.add_argument("--gen-backoff-initial", type=float, default=1.0)
    parser.add_argument("--gen-backoff-max", type=float, default=30.0)
    parser.add_argument("--gen-jitter", type=float, default=0.25)
    parser.add_argument("--gen-errors-path", default=DEFAULT_ERRORS)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in .env or environment.")

    client = OpenAI(api_key=api_key)
    config = BatchConfig(
        input_path=args.input,
        output_path=args.output,
        method="zero_shot",
        model=args.model,
        run_id=make_run_id(args.run_id or ""),
        start_index=args.start_index,
        max_questions=args.max_questions if args.max_questions is not None else -1,
        resume=args.resume,
        execute=args.execute,
        generation_retries=max(args.gen_retries, 0),
        generation_backoff_initial=max(args.gen_backoff_initial, 0.0),
        generation_backoff_max=max(args.gen_backoff_max, 0.0),
        generation_jitter=max(args.gen_jitter, 0.0),
        generation_errors_path=args.gen_errors_path or "",
    )
    if args.endpoint:
        config.endpoint = args.endpoint
    if args.timeout is not None:
        config.timeout = args.timeout
    if args.user_agent:
        config.user_agent = args.user_agent

    run_batch_generation(config=config, generate_fn=make_zero_shot_generator(
        client=client, model=args.model))


if __name__ == "__main__":
    main()
