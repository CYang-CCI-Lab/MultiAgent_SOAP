from tqdm import tqdm
import pandas as pd

def get_response(text):
    # ...existing get_response code...

# Add progress tracking
tqdm.pandas(desc="Processing summaries")
reference_df['Summary_expanded'] = reference_df['Summary'].progress_apply(
    lambda x: get_response(x) if pd.notnull(x) else x
)

# Display first few rows for verification
print("\nSample of processed data:")
print(reference_df[['Summary', 'Summary_expanded']].head())

# Optional: Display completion status
print(f"\nProcessed {len(reference_df)} rows")
