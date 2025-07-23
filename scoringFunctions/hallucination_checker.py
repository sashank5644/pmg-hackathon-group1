## John and Arjun
import openai
import re
from dotenv import load_dotenv
import os
import numpy as np

## load in api keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# Set up OpenAI client
client = openai.OpenAI(api_key=openai_api_key)  # Replace with your actual key
def score_responses_for_hallucination():
    prompt = "Who is the famous USA President Jeffery Dahmer?"
    responses = {
        "Response A": "The capital of Australia is Sydney.",
        "Response B": "He was never a president, but he was a serial killer.",
        "Response C": "He was the 54th President of the United States.",
        "Response D": "Jeffery Dahmer was a famous serial killer, not a president.",
        "Response E": "Jeffery Dahmer was a well-known serial killer, but he was never a president of the USA.",
        "Response F": "Jeffery Dahmer lost the presidential election in 1992.",
        "Response G": "Water is wet.",
        "Response H": "Jeffwey Danmer was not besident of Americuh",
    }
    system_prompt = (
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
        "100: Completely accurate and relevant"
    )
    results = {}
    for label, response_text in responses.items():
        user_prompt = f"""
Prompt:
{prompt}
Response:
{response_text}
Rate the factual accuracy of the response out of 100. Respond only with the number."""
        try:
            chat_response = client.chat.completions.create(
                model="o3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1
            )
            
            
            score_str = getattr(chat_response.choices[0].message, 'content', '').strip() \
            if getattr(chat_response, 'choices', None) else ''

            # Try to extract a number from the response
            results[label] = f"{score_str}/100"
        except Exception as e:
            print(f"[ERROR] Scoring failed for {label}: {e}")
       
    return results


# Run the example
def getHallucinationCheckerScore(prompt, responses, models):
    scores = score_responses_for_hallucination()
    for label, score in scores.items():
        print(f"{label}: {score}")
    return scores
 
