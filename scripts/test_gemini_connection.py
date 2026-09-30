import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["VLM_API_KEY"],
    base_url=os.environ["VLM_API_BASE_URL"],
)

response = client.chat.completions.create(
    model=os.environ["VLM_MODEL"],
    messages=[{"role": "user", "content": "Say hello in exactly five words."}]
)

print(response.choices[0].message.content)