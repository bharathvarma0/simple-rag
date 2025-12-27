
import os
from dotenv import load_dotenv
from openai import OpenAI
import sys

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in environment")
    sys.exit(1)

print(f"Testing access to model: gpt-5-nano")
print(f"API Key found: {api_key[:5]}...{api_key[-4:]}")

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("Success! Model is accessible.")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("\nFAILED to connect to gpt-5-nano.")
    print("Error details:")
    print(e)
