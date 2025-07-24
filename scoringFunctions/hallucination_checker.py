## John and Arjun
import openai
from dotenv import load_dotenv
import os

load_dotenv()
open_api_key = os.getenv("OPEN_AI_KEY")


client = openai.OpenAI(api_key=open_api_key)

openai_models = [
    "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o3-mini-high", "o3-pro",
    "o4-mini", "gpt-4.5", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4", "o1", "o1-pro", "o1-mini", "gpt-image-1", "dall-e-3", "sora"
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

    try:
        system_prompt = f"""
            "You will be given a prompt and a set of responses that an LLM could generate for that prompt. Please score each response based on its" \
            "factual accuracy. The score should be a number between 0 and 100, where 0 means completely inaccurate and 100 means completely accurate." \
            "Respond only with the number, without any additional text or explanation."
            "However, please keep in mind how the response related to the prompt. If the response is completely unrelated, it should receive a lower score." \
            "Also please keep in mind grammar and spelling mistakes, as they should negatively impact the score, if there are any spelling mistakes the score"
            "should not be higher than 70" \
            "Here's an example of how to view the range of scores:\n" \
            "0: Completely inaccurate or irrelevant\n" \
            "1-25: Very inaccurate and not relevant, or having terrible grammar and spelling\n" \
            "26-50: Possibly accurate, but containing information that is not relevant to the prompt\n" \
            "51-75: Partially accurate, but with significant errors, omissions, or spelling mistakes and bad grammar \n" \
            "76-90: Mostly accurate, with only minor errors or omissions\n" \
            "91-99: Almost completely accurate and relevant\n" \
            "100: Completely accurate and relevant"""

        user_prompt = f"""
            Prompt: 
            {prompt}

            Response:
            {response}

            Rate the factual accuracy of the response out of 100. Respond only with the number."""


        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10,
        )

        score_raw = completion.choices[0].message.content.strip()
        print(f"✅ Raw score received: '{score_raw}'")


        return score_raw

            # Try to extract a number from the response
            results[response_text] = score_str
        except Exception as e:
            print(f"[ERROR] Scoring failed: {e}")
       
    return results


# Run the example
def getHallucinationCheckerScore(prompt, responses, models):
    rawScores = getRawScores(prompt, responses, models)
    averageScores = getAverageScores(rawScores)
    return averageScores
    