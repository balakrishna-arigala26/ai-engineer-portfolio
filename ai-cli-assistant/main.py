import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from prompts import SYSTEM_PROMPT

# Load the .env file from the parent directory
load_dotenv(dotenv_path="../.env")


def main():
    print("AI CLI Assistant (Powered by Gemini)")
    print("Type 'exit' or 'quit' to end the session.\n")
    print("_" * 60)


    try:
        # Initialize the client
        client = genai.Client()

        # Configure the model with your custom System Prompt
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7, # A good balance between creativity and accuracy
        )

        # Start a chat session with the configuration applied
        chat = client.chats.create(
            model='gemini-2.5-flash-lite',
            config=config
        )

        # Start the conversation loop
        while True:
            user_input = input("\n🧑 You: ")

            if user_input.lower() in ['exit', 'quit']:
                print("\n🤖 AI: Goodbye! Happy coding. 👋")
                break

            if not user_input.strip():
                continue

            # Local file Reading
            if user_input.startswith('/read'):
                filepath = user_input.split('/read')[1].strip()
                try:
                    with open(filepath, 'r') as file:
                        content = file.read()
                    print(f"📄 Reading {filepath}...")
                    # Re-format the user input to include the file content
                    user_input = f"please analyze or summarize this file named {filepath}:\n\n{content}"
                except Exception as e:
                    print(f"\n❌ Could not read file: {e}")
                    continue

            # Error handling specifically for the API call   
            try:
                # Send message and get response
                response = chat.send_message(user_input)
                print(f"\n🤖 AI:\n{response.text}")
                print("_" * 60)
            except Exception as e:
                # Convert the error to a string so we can check what kind it is
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print("\n🤖 AI: Whoa there! I'm getting rate-limited by Google's Free Tier. Please wait a minute before asking another question.")
                    print("_" * 60)
                else:
                    print(f"\n❌ API Error: {e}")
                    print("_" * 60)



    except Exception as e:
        print(f"\n❌ A Critical startup error ocuurred: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🤖 AI: Session forcefully closed. Goodbye! 👋")
        sys.exit(0)
