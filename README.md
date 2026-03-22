# NL2SPARQL LLM Evaluation

This project evaluates large language models for generating SPARQL queries from natural language.

## Project Goals
- Compare GPT-based models and Code Llama
- Evaluate baseline vs RAG approaches
- Explore agentic workflows
- Generate SPARQL queries from natural language questions

## Project Structure
- pipelines/ → core pipelines (baseline, RAG, agentic)
- models/ → model-related code
- evaluation/ → evaluation scripts and metrics
- outputs/ → generated results
- prompts/ → prompt templates

## Setup

```bash
pip install -r requirements.txt