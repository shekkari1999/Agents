from embeddings import get_embeddings, cosine_similarity
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

documents = [
    "Python is a programming language",
    "Machine learning uses Python extensively", 
    "Cats are popular pets",
    "Deep learning is a subset of machine learning"
]
 
doc_embeddings = get_embeddings(documents)
 
results = vector_search("Artificial Intelligence", documents, doc_embeddings, top_k=4)
for r in results:
    print(f"{r['similarity']:.3f}: {r['chunk']}")