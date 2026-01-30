from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
def get_embeddings(texts, model="text-embedding-3-small"):
    """Convert text to embedding vectors."""
    if isinstance(texts, str):
        texts = [texts]
    
    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])
 
def vector_search(query, chunks, chunk_embeddings, top_k=3):
    """Find the most similar chunks to the query."""
    query_embedding = get_embeddings(query)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'similarity': similarities[idx]
        })
    return results

from sklearn.metrics.pairwise import cosine_similarity
 
sentences = [
    "The cat is sleeping on the couch",
    "A kitten is playing with a toy", 
    "The dog is running in the park"
]
embeddings = get_embeddings(sentences)
 
cat_kitten = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
cat_dog = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]