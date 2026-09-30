import base64
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["VLM_API_KEY"],
    base_url=os.environ["VLM_API_BASE_URL"],
)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_path = "data/raw/sroie/images/000.jpg"
b64_image = encode_image(image_path)

response = client.chat.completions.create(
    model=os.environ["VLM_MODEL"],
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What store is this receipt from, and what is the total amount?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ]
        }
    ]
)

print(response.choices[0].message.content)