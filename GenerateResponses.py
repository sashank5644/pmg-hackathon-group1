import openai
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

client = openai.OpenAI(api_key=api_key)  # instantiate the client (new SDK format)
anth_client = anthropic.Anthropic(api_key=anthropic_api_key)

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
        max_tokens=200,
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
        max_tokens=400,
    )
    return response.choices[0].message.content

def generateAnthropicPrompt(prompt):
    message = anth_client.messages.create(
        model="claude-opus-4-20250514",  # Use the correct up-to-date model name
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "rewrite the following prompt to make it better to put into an ai model:" + prompt}
        ]
    )
    return message.content[0].text

def generateAnthropicResponse(prompt):
    message = anth_client.messages.create(
        model="claude-opus-4-20250514",  # Use the correct up-to-date model name
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "You are a helpful AI assistant that answers questions or completes tasks in a clear, concise, and useful manner. Answer this prompt: " + prompt}
        ]
    )
    return message.content[0].text

def generateResponses(prompt):

    responses = [] 
    prompts = [] 

    # Step 1: generate possible responses from chatgpt 
    chatPrompt = generateChatPrompt(prompt)
    anthPrompt = generateAnthropicPrompt(prompt)
    prompts.append(prompt)
    prompts.append(chatPrompt)
    prompts.append(anthPrompt)


    for p in prompts:
        for _ in range(2):
            responses.append(generateChatResponse(p))
            responses.append(generateAnthropicResponse(p))
    

    # - claude returns 1 new prompt 

    #Step 2: have each model return responses based on these prompts
    # - chat 4.o returns 2 responses per prompt
    # - claude does the same 
    print("PROMPTS: ", prompts)
    return responses

# prompt = "Summarize the benefits of drinking water regularly."
# responses = generateResponses(prompt)

# print("\n--- ChatGPT Responses ---")
# for i, r in enumerate(responses, 1):
#     print(f"\nResponse {i}:\n{r}")
