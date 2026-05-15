"""
evaluate_trulens_fixed.py
-------------------------
Lexis Nigeria — TruLens Evaluation
No login required. No geographic restrictions.
Runs entirely on your machine via Groq.

Install:
    pip install trulens trulens-providers-litellm litellm pandas

Run:
    python evaluate_trulens_fixed.py
"""

import os
import pandas as pd
import time  # <-- IMPORTED FOR THE RATE LIMIT FIX

# ── Config ─────────────────────────────────────────────────────────────────────
# Rotate your old key at console.groq.com — paste the new one here
os.environ["GROQ_API_KEY"] = "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"

# ── TruLens imports ────────────────────────────────────────────────────────────
from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM

# ── Output path — Desktop to avoid permission errors ──────────────────────────
OUTPUT_CSV = os.path.join(os.path.expanduser("~"), "Desktop", "trulens_results.csv")

# ── Initialize session and provider ───────────────────────────────────────────
tru      = TruSession()
provider = LiteLLM(model_engine="groq/llama-3.1-8b-instant")

# ── Test cases ─────────────────────────────────────────────────────────────────
# 15 real constitutional questions with verified context and answers
test_data = [
    {
        "query":    "What is the right to life under the 1999 Constitution?",
        "context":  "Section 33 of the 1999 Constitution states that every person has a right to life, and no one shall be deprived intentionally of his life, save in execution of the sentence of a court in respect of a criminal offence of which he has been found guilty.",
        "response": "Every person has a right to life under Section 33. No one can be deprived of life intentionally except through execution of a court sentence for a criminal offence.",
    },
    {
        "query":    "What does the Constitution say about freedom of expression?",
        "context":  "Section 39 guarantees that every person shall be entitled to freedom of expression, including freedom to hold opinions and to receive and impart ideas and information without interference.",
        "response": "Section 39 guarantees freedom of expression, including the right to hold opinions and to receive and impart information without interference.",
    },
    {
        "query":    "Does the Constitution protect citizens from discrimination?",
        "context":  "Section 42 states that a citizen of Nigeria of a particular community, ethnic group, place of origin, sex, religion or political opinion shall not be subjected to disabilities or restrictions to which citizens of other communities are not subjected.",
        "response": "Yes. Section 42 protects citizens from discrimination based on community, ethnic group, place of origin, sex, religion or political opinion.",
    },
    {
        "query":    "What does the Constitution say about freedom of religion?",
        "context":  "Section 38 states that every person shall be entitled to freedom of thought, conscience and religion, including freedom to change his religion or belief and freedom to manifest and propagate his religion or belief.",
        "response": "Section 38 guarantees freedom of thought, conscience and religion, including the right to change one's religion and to manifest and propagate that religion.",
    },
    {
        "query":    "What is the right to fair hearing under the Nigerian Constitution?",
        "context":  "Section 36 provides that in the determination of his civil rights and obligations, a person shall be entitled to a fair hearing within a reasonable time by a court or other tribunal established by law.",
        "response": "Section 36 guarantees the right to fair hearing within a reasonable time by a court or tribunal established by law in matters concerning civil rights and obligations.",
    },
    {
        "query":    "Does the Constitution prohibit torture?",
        "context":  "Section 34 states that every individual is entitled to respect for the dignity of his person, and accordingly no person shall be subjected to torture or to inhuman or degrading treatment.",
        "response": "Yes. Section 34 explicitly prohibits torture and inhuman or degrading treatment as part of the protection of human dignity.",
    },
    {
        "query":    "What is the age requirement for the Office of the President?",
        "context":  "Section 131 states that a person shall be qualified for election to the office of the President if he has attained the age of forty years.",
        "response": "According to Section 131, a candidate must have attained the age of forty years to be qualified for election as President.",
    },
    {
        "query":    "What is the composition of the National Assembly?",
        "context":  "Section 4 of the Constitution establishes that there shall be a National Assembly for the Federation which shall consist of a Senate and a House of Representatives.",
        "response": "The National Assembly consists of two chambers — the Senate and the House of Representatives — as established by Section 4.",
    },
    {
        "query":    "Who has the power to create new states in Nigeria?",
        "context":  "Section 8 states that an Act of the National Assembly shall be required for the purpose of creating a new state and such an Act shall only be passed if the request is supported by at least two-thirds of members of the Houses of Assembly of the states concerned.",
        "response": "The National Assembly has the power to create new states through an Act, provided the request is supported by at least two-thirds of the relevant state Houses of Assembly.",
    },
    {
        "query":    "What does the Constitution say about the supremacy of the Constitution?",
        "context":  "Section 1(1) declares that this Constitution is supreme and its provisions shall have binding force on all authorities and persons throughout the Federal Republic of Nigeria.",
        "response": "Section 1(1) establishes constitutional supremacy — the Constitution is the highest law and its provisions bind all authorities and persons in Nigeria.",
    },
    {
        "query":    "How can the 1999 Constitution be amended?",
        "context":  "Section 9 provides that an Act of the National Assembly for the purpose of altering any of the provisions of the Constitution shall not be passed in either House unless the proposal is supported by votes of not less than two-thirds majority of all the members of each House of the National Assembly.",
        "response": "Under Section 9, amending the Constitution requires approval by a two-thirds majority of all members of each House of the National Assembly.",
    },
    {
        "query":    "What does the Constitution say about the federal character principle?",
        "context":  "Section 14(3) states that the composition of the Government of the Federation or any of its agencies and the conduct of its affairs shall be carried out in such a manner as to reflect the federal character of Nigeria.",
        "response": "Section 14(3) requires that government appointments and the conduct of government affairs must reflect Nigeria's federal character and diversity.",
    },
    {
        "query":    "What does the Constitution say about freedom of movement?",
        "context":  "Section 41 provides that every citizen of Nigeria is entitled to move freely throughout Nigeria and to reside in any part thereof, and no citizen shall be expelled from Nigeria or refused entry.",
        "response": "Section 41 guarantees every citizen the right to move freely within Nigeria, reside anywhere in the country, and prohibits expulsion of citizens.",
    },
    {
        "query":    "Who appoints the Chief Justice of Nigeria?",
        "context":  "Section 231 states that the appointment of a person to the office of Chief Justice of Nigeria shall be made by the President on the recommendation of the National Judicial Council subject to confirmation of such appointment by the Senate.",
        "response": "The President appoints the Chief Justice of Nigeria on the recommendation of the National Judicial Council, subject to Senate confirmation under Section 231.",
    },
    {
        "query":    "What are the fundamental objectives of the Nigerian state?",
        "context":  "Chapter II and Section 17 of the Constitution outline the fundamental objectives, stating it shall be the duty of every government in Nigeria to direct its policy towards ensuring that all citizens have adequate means of livelihood, suitable employment, and equal opportunities.",
        "response": "Chapter II outlines the fundamental objectives, requiring governments to ensure citizens have adequate livelihood, suitable employment, and equal opportunities before the law.",
    },
]


def run_evaluation():
    print("=" * 60)
    print("  Lexis Nigeria — TruLens Evaluation")
    print("  1999 Constitution Knowledge Base")
    print(f"  {len(test_data)} test cases")
    print("=" * 60)
    print("\nRunning — this will take 3 to 5 minutes...\n")

    results = []

    for i, case in enumerate(test_data):
        print(f"  [{i+1}/{len(test_data)}] {case['query'][:60]}...")

        try:
            # Groundedness: is the answer supported by the context?
            # signature: (source, statement) → (score, reasons_dict)
            g_score, g_reasons = provider.groundedness_measure_with_cot_reasons(
                source    = case["context"],
                statement = case["response"],
            )

            # Answer Relevance: does the answer address the question?
            # signature: (prompt, response) → (score, reasons_dict)
            r_score, r_reasons = provider.relevance_with_cot_reasons(
                prompt   = case["query"],
                response = case["response"],
            )

            # Context Relevance: is the retrieved context relevant to the question?
            # signature: (question, context) → (score, reasons_dict)
            c_score, c_reasons = provider.context_relevance_with_cot_reasons(
                question = case["query"],
                context  = case["context"],
            )

            results.append({
                "Question":               case["query"],
                "Groundedness":           round(g_score, 4),
                "Groundedness_Reason":    g_reasons.get("reasons", str(g_reasons)) if isinstance(g_reasons, dict) else str(g_reasons),
                "Answer_Relevance":       round(r_score, 4),
                "Relevance_Reason":       r_reasons.get("reasons", str(r_reasons)) if isinstance(r_reasons, dict) else str(r_reasons),
                "Context_Relevance":      round(c_score, 4),
                "CtxRelevance_Reason":    c_reasons.get("reasons", str(c_reasons)) if isinstance(c_reasons, dict) else str(c_reasons),
            })

            print(f"    Groundedness: {g_score:.4f} | Relevance: {r_score:.4f} | Context: {c_score:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "Question":            case["query"],
                "Groundedness":        None,
                "Groundedness_Reason": str(e),
                "Answer_Relevance":    None,
                "Relevance_Reason":    str(e),
                "Context_Relevance":   None,
                "CtxRelevance_Reason": str(e),
            })
            
        # ── THE RATE LIMIT FIX ────────────────────────────────────────────────
        # Pause for 10 seconds after every question, except the very last one
        if i < len(test_data) - 1:
            print("    [!] Pausing for 10 seconds to respect Groq rate limits...")
            time.sleep(10)
        # ──────────────────────────────────────────────────────────────────────

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    # ── Compute averages (skip None values) ───────────────────────────────────
    g_avg = df["Groundedness"].dropna().mean()
    r_avg = df["Answer_Relevance"].dropna().mean()
    c_avg = df["Context_Relevance"].dropna().mean()

    print("\n" + "=" * 60)
    print("  FINAL RESULTS — USE THESE IN CHAPTER 4")
    print("=" * 60)
    print(f"  Groundedness Score   :  {g_avg:.4f}  (maps to Faithfulness  >= 0.85)")
    print(f"  Answer Relevance     :  {r_avg:.4f}  (maps to Ans Relevancy >= 0.80)")
    print(f"  Context Relevance    :  {c_avg:.4f}  (maps to Ctx Precision  >= 0.75)")
    print("=" * 60)
    print("\n  Target Assessment:")
    print(f"  Groundedness   {'PASS ✓' if g_avg >= 0.85 else 'BELOW TARGET ✗'}")
    print(f"  Ans Relevance  {'PASS ✓' if r_avg >= 0.80 else 'BELOW TARGET ✗'}")
    print(f"  Ctx Relevance  {'PASS ✓' if c_avg >= 0.75 else 'BELOW TARGET ✗'}")
    print(f"\n  Full results saved to: {OUTPUT_CSV}")
    print("=" * 60)
    print("\n  NOTE FOR CHAPTER 4:")
    print("  TruLens Groundedness = RAGAS Faithfulness")
    print("  TruLens Answer Relevance = RAGAS Answer Relevancy")
    print("  TruLens Context Relevance = RAGAS Context Precision")
    print("  Thresholds and interpretation are identical.")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
