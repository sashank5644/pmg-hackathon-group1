import openai
from dotenv import load_dotenv
import os
import sys
from numpy import dot
from numpy.linalg import norm

# Allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GenerateResponses import generateResponses

# Load API key
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# Supported models
openai_models = [
    "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o3-mini-high", "o3-pro",
    "o4-mini", "gpt-4.5", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4", "o1", "o1-pro", "o1-mini", "gpt-image-1", "dall-e-3", "sora"
]

# Reverse-engineer the prompt
def getChatPrompt(response, model):
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

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def getSimilarity(inputPrompt, generatedPrompt):
    inputEmbedding = get_embedding(inputPrompt)
    generatedEmbedding = get_embedding(generatedPrompt)
    
    similarity = cosine_similarity(inputEmbedding, generatedEmbedding)

    return similarity

def chooseBestReverseModel(models):
    for preferred in ["gpt-4o", "gpt-4", "gpt-4.5", "o3"]:          #TODO: refine this based on best models for reverse engineering. 
        if preferred in models:
            return preferred, "open"
    model = models[0]
    if model in openai_models:
        return model, "open"
    else: 
        return model, "anth"

def getPossiblePrompts(responses, models):   
   
    prompts = {}
    model, api = chooseBestReverseModel(models)

    for response in responses:
        print("\n" + "=" * 80)
        print(f"🧠 Model Response:\n{response}\n")
        prompts[response] = {}

        if api == "open":
            reverse_prompt = getChatPrompt(response, model)
            prompts[response] = reverse_prompt
            print(f"🔁 Reverse-Engineered Prompt:\n{reverse_prompt}\n")

        print("=" * 80)

    return prompts 



# Evaluate and print results in a readable format
def getReverseEngineeringScore(originalPrompt, responses, models):
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


# Example usage
if __name__ == "__main__":
    prompt = "Elucidate the multifaceted and interdependent parameters influencing the efficacy of artificial intelligence in prognosticating clinical outcomes."
    responses = generateResponses(prompt)
    scores = getReverseEngineeringScore(prompt, responses, ["gpt-4o"])
