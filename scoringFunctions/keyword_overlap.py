import os
from dotenv import load_dotenv

# Install these packages:
# pip install openai anthropic
from openai import OpenAI
from anthropic import Client as AnthropicClient

# Load environment variables from .env
load_dotenv()


def init_client(provider: str):
    """
    Initialize and return a client for the given provider ('openai' or 'anthropic').
    """
    if provider.lower() == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        return OpenAI(api_key=key)

    elif provider.lower() == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment.")
        return AnthropicClient(api_key=key)

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_keywords_openai(client, text: str, top_k: int = 10) -> list[str]:
    """
    Use OpenAI LLM to extract top_k keywords from text.
    """
    prompt = (
        f"Extract the {top_k} most important keywords from the following text, "
        f"separated by commas, without any additional commentary:\n\n{text}"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    content = resp.choices[0].message.content
    # split on commas and strip whitespace
    keywords = [kw.strip() for kw in content.split(",") if kw.strip()]
    return keywords


def extract_keywords_anthropic(client, text: str, top_k: int = 10) -> list[str]:
    """
    Use Anthropic Claude to extract top_k keywords from text.
    """
    prompt = (
        f"Extract the {top_k} most important keywords from the following text, "
        f"separated by commas, without additional commentary:\n\n{text}"  
    )
    resp = client.completions.create(
        model="claude-2",   # or another Claude model
        prompt=client.HUMAN_PROMPT + prompt + client.AI_PROMPT,
        max_tokens=200,
    )
    content = resp.completion
    keywords = [kw.strip() for kw in content.split(",") if kw.strip()]
    return keywords


def compute_keyword_similarity(provider: str, prompt: str, responses: list[str], top_k: int = 10) -> dict[int, float]:
    """
    Extract keywords from prompt and each response using the specified LLM provider,
    then compute similarity scores (0-100) based on keyword overlap.

    Returns a dict mapping response index to similarity score.
    """
    client = init_client(provider)

    # pick extraction function
    if provider.lower() == "openai":
        extractor = extract_keywords_openai
    else:
        extractor = extract_keywords_anthropic

    # extract for prompt
    prompt_keys = set(extractor(client, prompt, top_k=top_k))
    print(f"Prompt keywords ({len(prompt_keys)}):", prompt_keys)

    scores: dict[int, float] = {}
    for idx, resp in enumerate(responses):
        resp_keys = set(extractor(client, resp, top_k=top_k))
        print(f"Response {idx} keywords ({len(resp_keys)}):", resp_keys)

        # compute overlap ratio
        if not prompt_keys or not resp_keys:
            score = 0.0
        else:
            intersection = prompt_keys.intersection(resp_keys)
            # similarity = overlap / prompt_keys_count * 100
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

    for provider in ["openai", "anthropic"]:
        print(f"\n--- Scores using {provider} ---")
        try:
            sim = compute_keyword_similarity(provider, prompt_text, sample_responses)
            print("Similarity scores:", sim)
        except Exception as e:
            print(f"Error with {provider}: {e}")
