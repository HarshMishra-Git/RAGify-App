import pandas as pd

def load_enterprise_data(filepath="data/sample_data.csv"):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filepath)
    # Ensure the 'text' column exists and return it as a list of texts
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column")
    return df['text'].tolist()

def preprocess_text(text):
    # Simple preprocessing (you can expand this)
    return text.strip().lower()  # Lowercase and strip extra spaces
