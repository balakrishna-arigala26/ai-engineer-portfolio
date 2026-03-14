# DevOps Log Analysis AI Agent

An automated Site Reliability Engineer (SRE) CLI tool that ingests raw server logs, pre-processes them for severity metrics, and leaverages a Large Language Model (LLM) to generate actionable root-cause analysis and recovery runbooks.

## Architecture

This project ytilizes a modular, microsrevice-style architecture to seprate file parsing from AI business logic:

* `main.py`: The CLI frontend. Handles user arguments, file ingestion, and terminal UI.
* `analyzer.py`: The AI backend. Uses Regex to generate a log severity dashboard, then feeds the raw log data into a highly constained LangChain/Gemini pipeline to extract exact Linux troubleshooting commands.

## Tech Stack

* **Language:** Python 3

* **Framework:** LangChain Core

* **Large Language Model:** Google Gemini (`gemini-2.5-flash-lite`)

* **Pre-Processing:** Python `re` (Regular Expressions) for case-insensitive anomaly detection.

## Features

1. **Severity Dashboard:** Automatically scans raw logs for case-insensitive failure states ( e.g., `fatal`, `panic`, `oom-kill`, `refused` ) and builds a visual metrics dashboard.
2. **Automated Runbooks:** Replaces vague AI suggestions with a strict SRE prompt that forces the output of exact, actionable Linux terminal commands ( e.g., `journalctl`, `ss -tulnp`, `top -o %MEM` ).
3. **Decoupled Logic:** The AI processing function accepts raw text strings, allowing it to be easily  ported into a Slack bo, FastAPI backend, or automated CI/CD pipeline in the future.

## How to Run

## 1. Setup Environment

Ensure you have a `.env` file in the parent directory conaining  your `GEMINI_API_KEY`.

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Analyze a Web Server Crash

```bash
python main.py sample_logs/nginx_error.log
```

## 4. Analyze a Database Kernel Panic

```bash
python main.py sample_logs/syslog_crash.log
```

## Example Output

When feeding a raw, chaotic Linux kernel panic ( OOM Killer ) log into the agent, it instantly generates this formatted triage report:


