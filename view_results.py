import pandas as pd
import os
from trulens.core import TruSession
from trulens.apps.basic import TruBasicApp

# 1. Initialize TruLens Session
tru = TruSession()

# 2. Define a dummy function to "replay" your questions
def dummy_app(question):
    return "Response loaded from CSV"

# 3. Initialize the recorder (Use the same app_id you used in your project)
recorder = TruBasicApp(dummy_app, app_name="Nigerian_Constitution_RAG", app_version="v1")

# 4. Load your results
# Adjust path if your CSV is in a different folder
csv_path = "trulens_results.csv" 
if not os.path.exists(csv_path):
    # Fallback to Desktop if that's where evaluate_trulens_fixed.py saved it
    csv_path = os.path.join(os.path.expanduser("~"), "Desktop", "trulens_results.csv")

df = pd.read_csv(csv_path)

print(f"Syncing {len(df)} records to the local database...")

for index, row in df.iterrows():
    # Run the dummy app to create a trace/record in the database
    with recorder as recording:
        recorder.main_call(row['Question'])
    
    # --- FIX: Retrieve the record object from the recording context ---
    record = recording.get() 
    record_id = record.record_id
    
    # 5. Inject your saved scores into the database
    # We use tru.add_feedback to manually link the CSV scores to the record
    try:
        tru.add_feedback(
            name="Groundedness", 
            record_id=record_id, 
            result=row['Groundedness']
        )
        tru.add_feedback(
            name="Answer Relevance", 
            record_id=record_id, 
            result=row['Relevance']
        )
        print(f"  Injecting record {index+1}: {row['Question'][:40]}...")
    except Exception as e:
        print(f"  Error injecting record {index+1}: {e}")

print("\nSuccess! Starting TruLens Dashboard...")
# 6. Launch the dashboard to see your Leaderboard
tru.run_dashboard()