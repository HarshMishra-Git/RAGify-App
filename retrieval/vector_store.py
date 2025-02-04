import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []

    def add_documents(self, embeddings, docs):
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(docs)

    def query(self, query_embedding, top_k=3):
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]
