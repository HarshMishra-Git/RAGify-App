from data_ingestion import load_enterprise_data, preprocess_text
from embeddings.embedding_utils import generate_embeddings, get_text_embedding
from retrieval.vector_store import VectorStore
from llm.llm_agent import generate_response

def build_rag_pipeline(data_filepath='data/sample_data.csv'):
    raw_texts = load_enterprise_data(data_filepath)
    documents = [preprocess_text(text) for text in raw_texts]

    embeddings = generate_embeddings(documents)
    embedding_dim = len(embeddings[0])

    vector_store = VectorStore(embedding_dim)
    vector_store.add_documents(embeddings, documents)

    return vector_store

def handle_query(query, vector_store):
    query_embedding = get_text_embedding(query)
    context_docs = vector_store.query(query_embedding, top_k=3)
    context = "\n".join(context_docs)
    return generate_response(query, context)
