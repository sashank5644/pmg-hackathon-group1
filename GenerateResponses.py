import openai
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

client = openai.OpenAI(api_key=api_key)  # instantiate the client (new SDK format)

def generateChatPrompt(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that rewords prompts to make them clearer, more engaging, or more effective for an AI model."
            },
            {
                "role": "user",
                "content": f"Rewrite the following prompt to make it better: {prompt}"
            }
        ],
        temperature=0.6,
        max_tokens=100,
    )
    return response.choices[0].message.content

def generateChatResponse(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions or completes tasks in a clear, concise, and useful manner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content

def generateAnthropicPrompt(prompt):
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    message = client.messages.create(
        model="claude-opus-4-20250514",  # Use the correct up-to-date model name
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

def generateAnthropicResponse(prompt):
    guardrails = "Generate a similar prompt based on the following input. Only return the prompt, numbered. Do not include any explanation, extra text before or after the responses."
    result = generateAnthropicPrompt(guardrails + prompt)

def generateResponses(prompt):

    responses = [] 
    prompts = [] 

    # Step 1: generate possible responses from chatgpt 
    chatPrompt = generateChatPrompt(prompt)
    prompts.append(prompt)
    prompts.append(chatPrompt)

    anthPrompt = generateAnthropicResponse(prompt)


    for p in prompts:
        for _ in range(2):
            responses.append(generateChatResponse(p))
    

    # - claude returns 1 new prompt 

    #Step 2: have each model return responses based on these prompts
    # - chat 4.o returns 2 responses per prompt
    # - claude does the same 
    return responses

prompt = "Summarize the benefits of drinking water regularly."
responses = generateResponses(prompt)

print("\n--- ChatGPT Responses ---")
for i, r in enumerate(responses, 1):
    print(f"\nResponse {i}:\n{r}")

print("anth" + generateAnthropicResponse(prompt))
