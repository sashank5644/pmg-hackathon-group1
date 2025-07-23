import os
from dotenv import load_dotenv

# Install these packages:
# pip install openai anthropic
from openai import OpenAI
import anthropic

# Load environment variables from .env
load_dotenv()

# Lists of supported models
openai_models = [
    "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o3-mini-high", "o3-pro",
    "o4-mini", "gpt-4.5", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4", "o1", "o1-pro", "o1-mini", "gpt-image-1", "dall-e-3", "sora"
]

anthropic_models = [
    "claude-opus-4-20250514", "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022"
]


def init_client(provider: str):
    """
    Initialize and return a client for the given provider ('openai' or 'anthropic').
    """
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        return OpenAI(api_key=key)

    elif provider == "anthropic":
        # Use the Anthropic Messages API
        key = os.getenv("ANTHROPIC_API_KEY")
        if key:
            return anthropic.Anthropic(api_key=key)
        return anthropic.Anthropic()

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_keywords_openai(client, text: str, top_k: int, model: str) -> list[str]:
    """
    Use OpenAI LLM to extract top_k keywords from text.
    """
    prompt = (
        f"Extract the {top_k} most important keywords from the following text, "
        f"separated by commas, without any additional commentary:\n\n{text}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    content = resp.choices[0].message.content
    return [kw.strip() for kw in content.split(",") if kw.strip()]


def extract_keywords_anthropic(client, text: str, top_k: int, model: str) -> list[str]:
    """
    Use Anthropic Claude Messages API to extract top_k keywords from text.
    """
    prompt = (
        f"Extract the {top_k} most important keywords from the following text, "
        f"separated by commas, without any additional commentary:\n\n{text}"
    )
    message = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens_to_sample=200,
        temperature=1,
    )
    content = message.content
    return [kw.strip() for kw in content.split(",") if kw.strip()]


def compute_keyword_similarity(model_name: str, prompt: str, responses: list[str], top_k: int = 10) -> dict[int, float]:
    """
    Determine provider by model_name, extract keywords, and compute similarity scores.

    Returns a dict mapping response index to similarity score (0–100).
    """
    # Identify provider
    if model_name in openai_models:
        provider = "openai"
    elif model_name in anthropic_models:
        provider = "anthropic"
    else:
        raise ValueError(f"Model '{model_name}' not supported for keyword extraction.")

    # Initialize client
    client = init_client(provider)

    # Choose extractor
    extractor = extract_keywords_openai if provider == "openai" else extract_keywords_anthropic

    # Extract keywords for prompt
    prompt_keys = set(extractor(client, prompt, top_k, model_name))
    print(f"Prompt keywords ({len(prompt_keys)}):", prompt_keys)

    scores: dict[int, float] = {}
    for idx, resp in enumerate(responses):
        resp_keys = set(extractor(client, resp, top_k, model_name))
        print(f"Response {idx} keywords ({len(resp_keys)}):", resp_keys)

        if not prompt_keys or not resp_keys:
            score = 0.0
        else:
            intersection = prompt_keys.intersection(resp_keys)
            score = len(intersection) / len(prompt_keys) * 100

        scores[idx] = round(score, 2)

    return scores


if __name__ == "__main__":
    prompt_text = "Explain the advantages of electric vehicles over traditional gasoline cars."
    sample_responses = [
        "Electric vehicles have lower operating costs because electricity is cheaper than gasoline.",
        "They produce zero tailpipe emissions, cutting air pollution and greenhouse gases.",
        "With regenerative braking, EVs can recapture energy and extend their driving range.",
        "Blockchain technology ensures secure, decentralized transaction records."
    ]

    # Example usage with specific models
    for model in ["o4-mini", "claude-opus-4-20250514"]:
        print(f"\n--- Scores using {model} ---")
        try:
            sim = compute_keyword_similarity(model, prompt_text, sample_responses)
            print("Similarity scores:", sim)
        except Exception as e:
            print(f"Error with {model}: {e}")
