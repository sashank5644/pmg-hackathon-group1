import openai
from dotenv import load_dotenv
import os
import sys



load_dotenv()
open_api_key = os.getenv("OPEN_AI_KEY")

open_client = openai.OpenAI(api_key=open_api_key)

openai_models = [
    "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o3-mini-high", "o3-pro",
    "o4-mini", "gpt-4.5", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4", "o1", "o1-pro", "o1-mini", "gpt-image-1"
]

noMaxTokens = [
    "o3", "o3-mini", "o3-pro", "o4-mini", "o1", "o1-mini"
]

def chatAnalyzeResponse(prompt, response, model):
    """
    Uses an LLM to score a candidate response based on relevance, accuracy, completeness, and clarity.

    Parameters:
        prompt (str): The original user prompt.
        response (str): The model-generated candidate response to evaluate.
        model (str): The name of the LLM judge to use (e.g., "gpt-4").
        open_client (OpenAI client): Initialized OpenAI API client.

    Returns:
        str: A numeric score (1–100) as a string, or an error message if the evaluation fails.
    """

    print(f"\n🔎 Evaluating response using model: {model}")
    print(f"📥 Prompt: {prompt}")
    print(f"📤 Response: {response}")
    try:
        evaluation_prompt = f"""
            You are an impartial and highly skilled evaluator of AI-generated answers.

            Your task is to grade the quality of the candidate's response to a user's prompt. You should consider the following criteria in your evaluation:

            1. **Relevance** – Does the response directly and clearly address the user's prompt?
            2. **Accuracy** – Is the information factual, truthful, and free from hallucinations or invented details?
            3. **Completeness** – Does the response fully answer the question or explain why it cannot be answered?
            4. **Clarity** – Is the response well-written, understandable, and logically structured?

            ---

            ⚠️ If the user's prompt contains **fabricated, fictional, or impossible claims** and the candidate response **correctly identifies this**, refuses to fabricate information, and explains the issue clearly — this is considered a **strong, high-quality answer**.

            ---

            Score the candidate from **1 to 100**, based on the overall quality and trustworthiness of the response:

            - **100**: Perfect — fully accurate, clear, and complete (or correctly refuses fictional premise with good explanation).
            - **90–99**: Excellent — highly accurate with only very minor omissions or flaws.
            - **70–89**: Good — mostly correct, but may lack completeness or clarity in places.
            - **40–69**: Mixed — partially helpful, vague, or with some factual issues.
            - **1–39**: Poor — mostly incorrect, misleading, confusing, or hallucinatory.

            Only output the number. Do not include any explanation.

            ---

            **Prompt:**  
            {prompt}

            **Candidate Response:**  
            {response}

            **Score (1–100):
        """


        if model not in noMaxTokens:

            completion = open_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an impartial LLM evaluator."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=10,
            )
        else:
            completion = open_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an impartial LLM evaluator."},
                    {"role": "user", "content": evaluation_prompt}
                ],
            )

        score_raw = completion.choices[0].message.content.strip()
        print(f"✅ Raw score received: '{score_raw}'")


        return score_raw

    except Exception as e:
        return f"[ERROR evaluating with model {model}]: {str(e)}"

def getAverageScores(scoreDict):
    """
    Calculates the average score for each response.
    """
    print("\n📊 Calculating average scores...")
    averages = {
        response: round(sum(scores)/len(scores), 2) if scores else None
        for response, scores in scoreDict.items()
    }

    for response, avg in averages.items():
        print(f"📈 Average score for response:\n{response[:60]}... -> {avg}")
    return averages

def getRawScores(prompt, responses, models): 
    collectiveScores = {}

    for response in responses: 
        collectiveScores[response] = []

        for model in models:
            if model in openai_models:
                scoreStr = chatAnalyzeResponse(prompt, response, model)
                try:
                    score = int(scoreStr)
                    if 1 <= score <= 100:
                        collectiveScores[response].append(score)
                        print(f"✅ Parsed score from {model}: {score}")
                    else:
                        print(f"[⚠️ Invalid score from {model}]:", scoreStr)
                except ValueError:
                    print(f"[⚠️ Could not parse score from {model}]:", scoreStr)

    return collectiveScores


def getLLMJudgeScore(prompt, responses, models):
    rawScores = getRawScores(prompt, responses, models)
    averageScores = getAverageScores(rawScores)
    return averageScores
    