from openai import OpenAI
from bertopic import BERTopic
import numpy as np
from scipy.special import rel_entr
from umap import UMAP
from hdbscan import HDBSCAN
import os
from dotenv import load_dotenv

def load_api_key() -> str:
    """Load OpenAI API key from a .env file."""
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("API key not found. Please set OPENAI_API_KEY in your .env file.")
    return key

# Initialize OpenAI client
api_key = load_api_key()
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
) -> tuple[dict[int, float], list[np.ndarray]]:
    """
    Embed prompt + responses, extract topic distributions via BERTopic,
    print keywords and raw probabilities, and return both similarity scores and raw distributions.
    """
    # 1) Embed texts
    resp = client.embeddings.create(
        input=[prompt] + responses,
        model=embedding_model
    )
    raw_embeddings = [d.embedding for d in resp.data]
    embeddings = np.array(raw_embeddings)
    docs = [prompt] + responses

    # 2) Configure dimensionality reduction and clustering
    umap_model = UMAP(init='random')
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1)

    # 3) Fit BERTopic model
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # 4) Print topic assignment and keywords
    for idx, topic in enumerate(topics):
        label = "prompt" if idx == 0 else f"response {idx-1}"
        if topic == -1:
            print(f"Document '{label}' is an outlier (no topic assigned).")
        else:
            kw_weights = topic_model.get_topic(topic)
            keywords = [word for word, _ in kw_weights]
            print(f"Document '{label}' assigned topic {topic} with keywords: {keywords}")

    # 5) Print raw topic-probabilities matrix
    matrix = np.stack(probs, axis=0)
    print("Raw topic-probabilities:\n", matrix)

    # 6) Compute Jensen–Shannon divergence scores
    prompt_dist = probs[0]
    sim_scores: dict[int, float] = {}
    for idx, dist in enumerate(probs[1:], start=0):
        jsd = jensen_shannon(prompt_dist, dist)
        # 1 – jsd gives 1.0 for identical, 0.0 for maximally different
        # multiply by 100 to map into [0,100]
        sim_scores[idx] = (1.0 - jsd) * 100
        
    return sim_scores, probs

if __name__ == "__main__":
    prompt = "Explain the advantages of electric vehicles over traditional gasoline cars."
    responses = [
        # These three should cluster together (all about EV benefits)
        "Electric vehicles have lower operating costs because electricity is cheaper than gasoline.",
        "They produce zero tailpipe emissions, cutting air pollution and greenhouse gases.",
        "With regenerative braking, EVs can recapture energy and extend their driving range.",
        # This one is off-topic and should be flagged as an outlier
        "Blockchain technology ensures secure, decentralized transaction records."
    ]

    similarity_scores, probs = compute_topic_similarity(prompt, responses)
    print("Similarity scores:", similarity_scores)