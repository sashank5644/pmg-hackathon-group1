from openai import OpenAI
from bertopic import BERTopic
import numpy as np
from scipy.special import rel_entr
from umap import UMAP
from hdbscan import HDBSCAN
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "API key not found. Please set OPENAI_API_KEY in your .env file."
    )

# Initialize OpenAI client
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
    print topic keywords for inspection, and return a dict mapping each response index to its JSD from the prompt.
    """
    # 1) Prepare docs and embed
    resp = client.embeddings.create(input=[prompt] + responses, model=embedding_model)
    raw_embeddings = [d.embedding for d in resp.data]
    embeddings = np.array(raw_embeddings)
    docs = [prompt] + responses

    # 2) Configure UMAP and HDBSCAN for small datasets
    umap_model = UMAP(init='random')
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1)

    # 3) Fit BERTopic
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # 4) Print detected topic and keywords for each document
    for idx, topic in enumerate(topics):
        label = "prompt" if idx == 0 else f"response {idx-1}"
        if topic == -1:
            print(f"Document '{label}' is an outlier (no topic assigned).")
        else:
            # Retrieve top keywords for this topic
            kw_weights = topic_model.get_topic(topic)
            keywords = [word for word, _ in kw_weights]
            print(f"Document '{label}' assigned topic {topic} with keywords: {keywords}")

    # 5) Compute JSD between prompt (probs[0]) and each response
    prompt_dist = probs[0]
    sim_scores: dict[int, float] = {}
    for idx, dist in enumerate(probs[1:], start=0):
        sim_scores[idx] = float(jensen_shannon(prompt_dist, dist))

    return sim_scores


if __name__ == "__main__":
    # Example usage
    prompt = "Explain the benefits of daily jogging for both physical and mental health."
    responses = [
        "Daily jogging strengthens your cardiovascular system and improves lung capacity.",
        "Regular running releases endorphins, reducing stress and boosting mood.",
        "Blockchain technology secures transactions through a decentralized ledger."
    ]

    similarity_scores = compute_topic_similarity(prompt, responses)
    print("Similarity scores:", similarity_scores)
