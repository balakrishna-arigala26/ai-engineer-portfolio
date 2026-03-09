# AI CLI Assistant


A modular, terminal-based AI Assistant powered by Python and the Google Gemini API. This project goes beyond standard API calls by implementing stateful memory, system prompting, and local file analysis.


## Feartures
* **Stateful Memory:** Maintains conversation history for seamless follow-up questions.
* **System Prompting:** Uses a dedicated `prompts.py` file to strictly contol the AI's persona and ensure technical, concise outputs.
* **Local File Analysis:** Features a custom `/read <filename>` command, allowing the AI to ingest and summarize local files directly  from terminal.
* **Resilient Error Handling:** Includes custom exception handling to grcefully manage API rate limits (like Fre Tier exhaustion) without crashing.


## Tech Stack
* Python
* Google GenAI SDK (`google-genai`)
* Model: `gemini-2.5-flash-lite`


## How to Run
1. Clone the repository and navigate to this folder.
2. Ensure you have a `.env` file in the parent directory with your `GEMINI_API_KEY`.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the assistant: `python main.py`