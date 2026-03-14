import os
import sys
from analyzer import analyze_log


if __name__ == "__main__":
    print("🚀 DevOps Log Analysis AI Agent\n")

    # Check if user provided a file argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_log_file>")
        sys.exit(1)


    log_file = sys.argv[1]


    # Check if file exists
    if not os.path.exists(log_file):
        print(f"❌ Error: Log file '{log_file}' not found.")
        sys.exit(1)

    
    print(f"📂 Ingesting log file: {log_file}...")
    with open(log_file, 'r') as file:
        log_content = file.read()


    print("⏳ Analyzing stack trace....\n")
    result = analyze_log(log_content)

    # Print the final report
    print("=" * 60)
    print(result)
    print("=" * 60)
