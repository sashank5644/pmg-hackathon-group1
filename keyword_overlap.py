from openai import OpenAI
from bertopic import BERTopic
import numpy as np
from scipy.special import rel_entr

api_key = ""
client = OpenAI(api_key=api_key)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the Jensen–Shannon divergence between two probability distributions.
    """
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(rel_entr(p, m)) + np.sum(rel_entr(q, m)))

def compute_topic_similarity(
    prompt: str,
    responses: list[str],
    embedding_model: str = "text-embedding-ada-002"
) -> dict[int, float]:
    """
    Embed prompt + responses, extract topic distributions via BERTopic,
    and return a dict mapping each response index to its JSD from the prompt.
    
    Args:
      prompt:     The user’s prompt text.
      responses:  A list of generated-response strings.
      embedding_model: OpenAI embedding model to call.
    
    Returns:
      {i: score} where i is the index of responses[i], and
      score is the Jensen–Shannon divergence to the prompt’s topic mix.
    """
    # 1) Prepare docs and embed
    resp = client.embeddings.create(input=[prompt] + responses, model=embedding_model)
    embeddings = [d.embedding for d in resp.data]
    docs = [prompt] + responses
    # 2) Fit BERTopic to get per-doc topic distributions
    topic_model = BERTopic()
    _, probs = topic_model.fit_transform(docs, embeddings)
    # probs is a list of numpy arrays, each summing to 1
    
    # 3) Compute JSD between prompt (probs[0]) and each response
    prompt_dist = probs[0]
    sim_scores: dict[int, float] = {}
    for idx, dist in enumerate(probs[1:], start=0):
        sim_scores[idx] = float(jensen_shannon(prompt_dist, dist))
    
    return sim_scores


if __name__ == "__main__":
    # Example usage
    prompt = "What are the benefits of using renewable energy sources?"
    responses = [
        "Renewable energy sources like solar and wind power reduce greenhouse gas emissions.",
        "Using fossil fuels is cheaper than renewable energy sources.",
        "Solar panels can significantly lower electricity bills."
    ]
    
    similarity_scores = compute_topic_similarity(prompt, responses)
    print(similarity_scores)

