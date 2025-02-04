# import openai
# import numpy as np

# # Set your OpenAI API key
# openai.api_key = "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"

# def get_text_embedding(text, model="text-embedding-ada-002"):
#     response = openai.embeddings.create(
#         input=[text],
#         model=model
#     )
#     embedding = response['data'][0]['embedding']
#     return np.array(embedding)

# # Batch processing utility
# def generate_embeddings(texts):
#     embeddings = []
#     for text in texts:
#         embedding = get_text_embedding(text)
#         embeddings.append(embedding)
#     return embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the Sentence Transformer model (LLaMA or another available embedding model)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Change model if needed

def get_text_embedding(text):
    """Generate embedding for a given text."""
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

def generate_embeddings(texts):
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings
