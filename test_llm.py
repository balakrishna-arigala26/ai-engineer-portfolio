# For openai
""" # from openai import OpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# client = OpenAI()


# response = client.chat.completions.create(
# 	model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "Explain AI agents in simple terms"}],
# )

# print(response.choices[0].message.content)"""


# for Google-AI

import os
from dotenv import load_dotenv
from google import genai

# Load API key from the .env file
load_dotenv()

# Initialize the new Gemini client.
# It automatically finds the GOOGLE_API_KEY loaded by dotenv!
client = genai.Client()

# Generate a response using the new syntax
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents="Explain AI agents in simple terms."
)

# Print the result
print(response.text)
