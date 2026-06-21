from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen2.5-72B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who is the current Pope?"},
    ]
)
print("==" * 20)
print("Model: Qwen2.5-72B")
print("==" * 20)
print("System Message: You are a helpful assistant.")
print("User Question: Who is the current Pope?")
print("Chat response:", chat_response.choices[0].message.content)