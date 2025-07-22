## John and Arjun
import openai
import re
from dotenv import load_dotenv
import os   

## load in api keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# Set up OpenAI client
client = openai.OpenAI(api_key=openai_api_key)  # Replace with your actual key
def score_responses_for_hallucination():
    prompt = "What is the capital of Australia?"
    reference = "The capital city of Australia is Canberra."
    responses = {
        "Response A": "The capital of Australia is Sydney.",
        "Response B": "The capital of Australia is Canberra.",
        "Response C": "Australia has no official capital, but Sydney is often considered the capital.",
        "Response D": "Melbourne was the capital until 1988, but now it's Sydney."
    }
    system_prompt = (
        "You are a fact-checking assistant. Score the factual accuracy of a single response on a scale of 0 to 100, "
        "where 100 means the response has no hallucinations and is fully supported by the reference or prompt, "
        "and lower scores indicate more hallucinations. "
        "Respond only with the number (e.g., '87')."
    )
    results = {}
    for label, response_text in responses.items():
        user_prompt = f"""Reference:
{reference}
Prompt:
{prompt}
Response:
{response_text}
Rate the factual accuracy of the response out of 100. Respond only with the number."""
        try:
            chat_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            
            

            score_str = getattr(chat_response.choices[0].message, 'content', '').strip() \
            if getattr(chat_response, 'choices', None) else ''

            # Try to extract a number from the response
            match = re.search(r'\b(\d{1,3})\b', score_str)
            if match:
                score = min(max(int(match.group(1)), 0), 100)  # Clamp between 0–100
            else:
                print(f"[WARN] Could not parse score for {label}, defaulting to 50. Raw response: {score_str}")
                score = 50
        except Exception as e:
            print(f"[ERROR] Scoring failed for {label}: {e}")
            score = 50
        results[label] = f"{score}/100"
    return results
# Run the example
if __name__ == "__main__":
    scores = score_responses_for_hallucination()
    print("\nHallucination Scores:")
    for label, score in scores.items():
        print(f"{label}: {score}")
