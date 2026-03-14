import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the monorepo .env file
load_dotenv(dotenv_path="../.env")


def summarize_logs(log_text):
    # Case-insensitive regex for real-world DevOps keywords
    error_count = len(re.findall(r"(?i)(error | failed | fatal | panic | killed | oom-kill | critical | refused )", log_text))
    warn_count = len(re.findall(r"(?i)(warnings | warn | retry | timeout)", log_text))
    info_count = len(re.findall(r"(?i)(info | notice | starting | terminated | exited)", log_text))

    summary = (
        "📊 LOG SEVERITY SUMMARY\n"
        f"🔴 CRITICAL/ERRORS : {error_count}\n"
        f"🟡 WARNINGS        : {warn_count}\n"
        f"🟢 INFO            : {info_count}\n"
    )
    return summary

def analyze_log(log_text):
    print("🧠 Initializing Senior SRE Agent...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # 1. Generate the pre-p;rocessing metrics dashboard
    summary_dashboard = summarize_logs(log_text)
    # 2. The Strict SRE Prompt
    prompt = f"""
    You are a Senior Linux DevOps Engineer and Site Reliability Engineer (SRE). 
    Your job is to analyze the following server log, diagnose the exact failure, and provide the fix. 

    Format your response EXACTLY like this:
    🚨 ROOT CAUSE: (1 clear sentence explaining the failure)
    💡 EXPLANATION: (A brief, simple explanation of what the specific error codes mean) 
    🛠️ COMMANDS TO FIX: (Provide the exact Linux terminal commands to troubleshoot or fix the issue)

    Here is the raw log data:
    ---
    {log_text}
    ---
    """ 

    # 3. Call the Gemini AI 
    response = llm.invoke(prompt) 

    # 4. Merge the Metrics Dashboard with the AI Analysis 
    final_output = f"{summary_dashboard}\n🤖 AI ROOT CAUSE ANALYSIS\n{'_'*30}\n\n{response.content}" 
    return final_output