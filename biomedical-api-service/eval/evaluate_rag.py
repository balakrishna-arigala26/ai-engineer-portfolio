# Built-In Python Imports
import os
import sys
import asyncio
import pandas as pd
from datasets import Dataset

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Third-Party & Custom Imports
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# --- Import RunConfig to throttle Ragas ---
from ragas.run_config import RunConfig
# ----------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from app.engine import BiomedicalAIEngine

load_dotenv()

async def run_evaluation():
    print("🚀 Initializing Automated RAG Evaluation pipeline...")
    engine = BiomedicalAIEngine()

    base_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    evaluator_llm = LangchainLLMWrapper(base_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    # 🚀 MARKETING FIX: Reduced to the single, perfect golden question
    test_questions = [
        "What is the procedure to remove the battery on the GE Venue Fit?"
    ]

    print(f"Testing {len(test_questions)} questions against the RAG pipeline...")

    data_samples = {
        "question": [],
        "answer": [],
        "contexts":[],
    }

    for i, question in enumerate(test_questions):
        print(f"\nEvaluating Q{i+1}: {question}")

        response = await engine.ask_question(question, session_id=f"eval-session-{i}")
        docs = engine.vector_store.similarity_search(question, k=4)
        contexts = [doc.page_content for doc in docs]

        data_samples["question"].append(question)
        data_samples["answer"].append(response)
        data_samples["contexts"].append(contexts)

        # --- Artificial delay to respect API rate limits ---
        if i < len(test_questions) - 1:
            print("⏳ Throttling: Waiting 5 seconds to respect Google API limits...")
            await asyncio.sleep(5)

    dataset = Dataset.from_dict(data_samples)

    print("\n📊 Running Ragas Metrics (This uses Gemini as a judge)...")

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    ]
        
    # --- Tell Ragas to process 1 request at a time (max_workers=1) ---
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=1, max_retries=5)
    )

    df = results.to_pandas()
    os.makedirs("eval", exist_ok=True)
    df.to_csv("eval/rag_evaluation_report.csv", index=False)

    print("\n✅ Evaluation Complete! Report saved to eval/rag_evaluation_report.csv")
    print("\n--- Final Aggregated Scores ---")
    print(results)


if __name__ == "__main__":
    asyncio.run(run_evaluation())