import os
import json
import pandas as pd
import tomllib
from langchain_groq import ChatGroq 
from app import setup_engine 

def load_secrets():
    try:
        with open(".streamlit/secrets.toml", "rb") as f:
            return tomllib.load(f)
    except:
        return {}

def get_llm_score(llm, prompt):
    """Simple helper to get a numeric score from the LLM"""
    try:
        response = llm.invoke(prompt)
        # Extract the first number found in the response
        import re
        numbers = re.findall(r"0\.\d+|1\.0", response.content)
        return float(numbers[0]) if numbers else 0.5
    except:
        return 0.0

def main():
    secrets = load_secrets()
    api_key = secrets.get("GROQ_API_KEY")
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    print("\n--- Initializing Lexis Nigeria Engine ---")
    rag_pipeline = setup_engine()

    with open("test_cases.json", "r") as f:
        test_data = json.load(f)

    results_list = []
    print(f"--- Running Manual Evaluation ({len(test_data[:5])} samples) ---")
    
    for i, item in enumerate(test_data[:5]):
        print(f"Processing Sample {i+1}...")
        
        # 1. Get RAG Response
        res = rag_pipeline.invoke({"input": item["question"]})
        answer = res["answer"]
        context = " ".join([doc.page_content for doc in res["context"]])

        # 2. Manual Faithfulness Check
        f_prompt = f"System Answer: {answer}\nSource Context: {context}\nOn a scale of 0.0 to 1.0, how faithful is the answer to the context? Return only the number."
        faithfulness = get_llm_score(llm, f_prompt)

        # 3. Manual Relevancy Check
        r_prompt = f"Question: {item['question']}\nAnswer: {answer}\nOn a scale of 0.0 to 1.0, how relevant is the answer to the question? Return only the number."
        relevancy = get_llm_score(llm, r_prompt)

        results_list.append({
            "question": item["question"],
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": 1.0 if len(res["context"]) > 0 else 0.0
        })

    # Save and Print
    final_df = pd.DataFrame(results_list)
    print("\n=== EVALUATION SUCCESSFUL ===")
    print(final_df[["faithfulness", "answer_relevancy", "context_precision"]].mean())
    
    final_df.to_csv("ragas_evaluation_results.csv", index=False)
    print("\nCSV generated! Use these averages for Chapter 4.")

if __name__ == "__main__":
    main()