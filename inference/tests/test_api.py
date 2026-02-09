import argparse
from model import query_openai, query_claude, query_gemini, AVAILABLE_MODELS
from rich.console import Console

console = Console()


def query_llm(args: argparse.Namespace, prompt: str) -> str:
    """
    Query LLM model with arguments and user prompt

    Args:
    - args: argparse.Namespace
    - prompt: str

    Returns:
    - response: str
    """
    model = args.model
    if "gpt-3.5-turbo" in model or "gpt-4" in model:
        return query_openai(args, prompt, tok, console)
    elif "claude" in model:
        return query_claude(args, prompt, console)
    elif model == "gemini-pro":
        return query_gemini(args, prompt, console)
    else:
        # use openai pathways as default
        return query_openai(args, prompt, tok, console)
