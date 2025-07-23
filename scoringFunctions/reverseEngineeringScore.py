import openai
import anthropic
from dotenv import load_dotenv
import os
import sys
from numpy import dot
from numpy.linalg import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generatePotentialResponses.GenerateResponses import generateResponses

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

client = openai.OpenAI(api_key=api_key)
anth_client = anthropic.Anthropic(api_key=anthropic_api_key)

# Supported OpenAI models
openai_models = [
    "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o3-mini-high", "o3-pro",
    "o4-mini", "gpt-4.5", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4", "o1", "o1-pro", "o1-mini", "gpt-image-1", "dall-e-3", "sora"
]

anthropic_models = [
    "claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"
]

def getChatPrompt(response, model):
    """
    Uses a language model to reverse-engineer the likely user prompt that could have produced a given response. 

    Parameters: 
        response (str): the AI-generated response to analyze.
        model (str): Model to use for the reverse-prompting. 

    Returns: 
        str: a guessed version of the original prompt. 
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes model responses and tries to guess the original user prompt that could have generated the response."
                },
                {
                    "role": "user",
                    "content": f"""Given the following AI response, guess what the original user prompt might have been.
                    Respond with only the guessed prompt — do not include any introduction or explanation. Do not include any quotations around the prompt. 

                    AI response:
                    {response}"""
                }
            ],
            temperature=0.6,
            max_tokens=100,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] with model {model}]: {str(e)}"

def getAnthPrompt(response, model):
    """
    Uses a Claude (Anthropic) model to reverse-engineer a prompt from an AI-generated response.

    Parameters:
        response (str): The AI-generated response to analyze.
        model (str): The Claude model to use.

    Returns:
        str: Guessed original user prompt.
    """
    try:
        message = anth_client.messages.create(
            model=model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": f"""Given the following AI response, guess what the original user prompt might have been.
                    Respond with only the guessed prompt — do not include any introduction or explanation. Do not include any quotations around the prompt.
                    AI response{response}"""
                }
            ]
        )
        return message.content[0].text.strip()
    except Exception as e:
        return f"[ERROR with model {model}]: {str(e)}"


def get_embedding(text, model="text-embedding-3-small"):
    """
    Converts a string into a numerical vector embedding using OpenAI's embedding model. 

    Parameters: 
        text (str): The text to embed
        model (str): The embedding model to use.

    Returns:
        list[float]: the embedding vector. 
    """
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Parameters: 
        vec1, vec2 (list[float]): Input vectors.

    Returns:
        float: Cosine similarity score between -1 and 1. 
    """
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def getSimilarity(inputPrompt, generatedPrompt):
    """
    Calculates the semantic similarity between the original and the reversed-engineered prompt. 

    Parameters:
        inputPrompt (str)" Original user prompt.
        generatedPrompt (str): Reverse-engineered prompt.
    
    Returns:
        float: Cosine similarity between their embeddings. 
    """
    inputEmbedding = get_embedding(inputPrompt)
    generatedEmbedding = get_embedding(generatedPrompt)
    
    similarity = cosine_similarity(inputEmbedding, generatedEmbedding)

    return similarity

def chooseBestReverseModel(models):
    """
    Chooses the best available model for reverse-prompting based on preference. 

    Parameters:
        models (list[str]): List of available model names.
    
    Returns: 
        tuple: (selected_model, "open" or "anthropic" string to indicate provider)
    """
    for preferred in ["gpt-4o", "gpt-4", "gpt-4.5", "o3"]:          #TODO: refine this based on best models for reverse engineering. 
        if preferred in models:
            return preferred, "open"
    model = models[0]
    if model in openai_models:
        return model, "open"
    else: 
        return model, "anth"

def getPossiblePrompts(responses, models):   
    """
    Applies reverse-prompting to a list of model responses. 

    Parameters:
        responses (list[str]): List of AI-generated responses.
        models (list[str]): List of model names to choose from for reverse prompting. 
    
    Returns: 
        dict: Maps each response to its reverse-engineered prompt. 
    """
    prompts = {}
    model, api = chooseBestReverseModel(models)

    for response in responses:
        print("\n" + "=" * 80)
        print(f"🧠 Model Response:\n{response}\n")
        prompts[response] = {}

        if api == "open":
            reverse_prompt = getChatPrompt(response, model)
            print(f"🔁 Reverse-Engineered Prompt:\n{reverse_prompt}\n")
        elif api == "anth":
            reverse_prompt = getAnthPrompt(response, model)
            print(f"🔁 Reverse-Engineered Prompt:\n{reverse_prompt}\n")
        else:
            reverse_prompt = "[ERROR] No valid provider"
        prompts[response] = reverse_prompt

        print("=" * 80)

    return prompts 


def getReverseEngineeringScore(originalPrompt, responses, models):
    """
    Computes similarity scores bewteen the original prompt and reverse-engineered prompts
    for a list of AI-generated responses.

    Parameters:
        originalPrompt (str): the initial user prompt. 
        responses (list[str]): Responses generated by different LLMs. 
        models (list[str]): Available model names for reverse engineering. 

    Returns:
        dict: Maps each response to a scaled similarity score (0-100).
    """
    prompts = getPossiblePrompts(responses, models)
    scores = {}

    for response, reversePrompt in prompts.items():
        if "[ERROR]" in reversePrompt:
            similarity = 0.0
        else:
            similarity = getSimilarity(originalPrompt, reversePrompt)
            scores[response] = ((round(similarity, 4) + 1) / 2) * 100 
            print("prompt: ", reversePrompt)
            print("score:", scores[response])
    return scores


# # Example usage
# if __name__ == "__main__":
#     prompt = "List five peer-reviewed papers proving that unicorns existed, including DOIs."
#     responses = generateResponses(prompt)
#     scores = getReverseEngineeringScore(prompt, responses, ["gpt-4o"])
