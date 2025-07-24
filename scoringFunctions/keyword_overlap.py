import os
from dotenv import load_dotenv

# Install these packages:
# pip install openai anthropic
from openai import OpenAI
from anthropic import Anthropic

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
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        return OpenAI(api_key=key)

    elif provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment.")
        return Anthropic(api_key=key)

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_keywords_openai(client, text: str, top_k: int, model: str) -> list[str]:
    prompt = (
        f"Identify the {top_k} most important topic words that appear exactly as written in the following text. "
        f"Only return words that are explicitly present in the text. List them as comma-separated keywords with no extra explanation:\n\n{text}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0
    )
    content = resp.choices[0].message.content
    return [kw.strip() for kw in content.split(",") if kw.strip()]


def extract_keywords_anthropic(client, text: str, top_k: int, model: str) -> list[str]:
    prompt = (
        f"Identify the {top_k} most important topic words that appear exactly as written in the following text. "
        f"Only return words that are explicitly present in the text. List them as comma-separated keywords with no extra explanation:\n\n{text}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.content[0].text.strip() if hasattr(response.content[0], 'text') else response.content
    return [kw.strip() for kw in content.split(",") if kw.strip()]


def score_keywords_with_llm(client, prompt_keywords: list[str], response_keywords: list[str], model: str, provider: str) -> float:
    prompt = (
        f"Based on the following sets of keywords, give a relevance score from 0 to 100 indicating how related the response keywords are to the prompt keywords.\n"
        f"Prompt Keywords: {', '.join(prompt_keywords)}\n"
        f"Response Keywords: {', '.join(response_keywords)}\n"
        f"Just return a single numeric score."
    )

    if provider == "openai":
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        content = resp.choices[0].message.content.strip()
    else:
        response = client.messages.create(
            model=model,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip() if hasattr(response.content[0], 'text') else response.content

    try:
        return round(float(content))
    except:
        return 0.0


def compute_keyword_similarity(
    model_name: str,
    prompt: str,
    responses: list[str],
    top_k: int = 10
) -> dict[int, float]:
    if model_name in openai_models:
        provider = "openai"
    elif model_name in anthropic_models:
        provider = "anthropic"
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    client = init_client(provider)
    extractor = extract_keywords_openai if provider == "openai" else extract_keywords_anthropic

    prompt_keys = extractor(client, prompt, top_k, model_name)
    print(f"Prompt keywords ({len(prompt_keys)}): {set(prompt_keys)}")

    scores: dict[int, float] = {}
    for idx, resp in enumerate(responses):
        resp_keys = extractor(client, resp, top_k, model_name)
        print(f"Response {idx} keywords ({len(resp_keys)}): {set(resp_keys)}")

        score = score_keywords_with_llm(client, prompt_keys, resp_keys, model_name, provider)
        scores[idx] = score

    return scores


if __name__ == "__main__":
    prompt_text = "Explain the advantages of electric vehicles over traditional gasoline cars."
    sample_responses = [
        "Electric vehicles have lower operating costs because electricity is cheaper than gasoline.",
        "They produce zero tailpipe emissions, cutting air pollution and greenhouse gases.",
        "With regenerative braking, EVs can recapture energy and extend their driving range.",
        "Blockchain technology ensures secure, decentralized transaction records."
    ]

    for model in ["gpt-4o", "claude-opus-4-20250514"]:
        print(f"\n--- Scores using {model} ---")
        try:
            sim = compute_keyword_similarity(model, prompt_text, sample_responses)
            print("Similarity scores:", sim)
        except Exception as e:
            print(f"Error with {model}: {e}")