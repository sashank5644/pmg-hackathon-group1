import openai
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

client = openai.OpenAI(api_key=api_key)  # instantiate the client (new SDK format)
anth_client = anthropic.Anthropic(api_key=anthropic_api_key)

anthropic_models = [
    "claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"
]
openai_models = [
    "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o3-mini-high", "o3-pro",
    "o4-mini", "gpt-4.5", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4", "o1", "o1-pro", "o1-mini", "gpt-image-1", "dall-e-3", "sora"
]

def generateChatPrompt(prompt, model):
    """
    Uses GPT-4 to rewrite a given prompt to be more clear, engaging, or effective.

    Parameters:
        prompt (str): The original user input prompt. 

    Returns:
        str: Rewritten version of the prompt. 
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that rewords prompts to make them clearer, more engaging, or more effective for an AI model."
            },
            {
                "role": "user",
                "content": f"Rewrite the following prompt to make it better: {prompt} Respond with only the guessed prompt — do not include any introduction or explanation. Do not include any quotations around the prompt. "
            }
        ],
        temperature=0.6,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def generateChatResponse(prompt, model):
    """
    Uses GPT-4 to generate a response to a user prompt.

    Parameters:
        prompt (str): The prompt to respond to. 

    Returns:
        str: Generated model response. 
    """
    response = client.chat.completions.create(
        model=model,
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
        max_tokens=1024,
    )
    return response.choices[0].message.content

def generateAnthropicPrompt(prompt, model):
    """
    Uses Claude to rewrite a prompt for better clarity or engagement. 

    Parameters: 
        prompt (str): The original user input prompt. 

    Returns:
        str: Rewritten prompt by Claude.
    """
    message = anth_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Rewrite the following prompt to make it better:" + prompt + "Respond with only the guessed prompt — do not include any introduction or explanation. Do not include any quotations around the prompt. "}
        ]
    )
    return message.content[0].text

def generateAnthropicResponse(prompt, model):
    """
    Uses Claude (Anthropic) to generate a response to a user prompt.

    Parameters: 
        prompt (str)" The prompt to respond to.

    Returns:
        str: Model-generated response. 
    """
    message = anth_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "You are a helpful assistant that answers questions or completes tasks in a clear, concise, and useful manner. Answer this prompt: " + prompt}
        ]
    ) 
    return message.content[0].text

def generateResponses(prompt, models):
    responses = []
    prompts = [prompt]  # Start with original prompt

    for model in models:
        try:
            if model in openai_models:
                rewritten = generateChatPrompt(prompt, model)
            elif model in anthropic_models:
                rewritten = generateAnthropicPrompt(prompt, model)
            else:
                print(f"[WARNING] Unknown model '{model}' — skipping.")
                continue

            prompts.append(rewritten)
        except Exception as e:
            print(f"[ERROR] Prompt generation failed for model {model}: {e}")
            prompts.append(prompt)

    for p in prompts:
        for model in models:
            try:
                if model in openai_models:
                    response = generateChatResponse(p, model)
                elif model in anthropic_models:
                    response = generateAnthropicResponse(p, model)
                else:
                    continue

                responses.append(response)
            except Exception as e:
                print(f"[ERROR] Response generation failed for model {model}: {e}")

    print("PROMPTS:", prompts)
    return responses


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
