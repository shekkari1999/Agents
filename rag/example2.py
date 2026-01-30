from tavily import TavilyClient
import os 
from dotenv import load_dotenv
from chunking import fixed_length_chunking
from embeddings import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


response = tavily.search(
    "2025 Nobel Prize winners",
    max_results=10,
    include_raw_content=True
)
 
search_results = []
for result in response['results']:
    if result.get('raw_content'):
        search_results.append({
            'title': result['title'],
            'content': result['raw_content'],
            'url': result['url']
        })
all_chunks = []
for result in search_results:
    text = f"Title: {result['title']}\n{result['content']}"
    chunks = fixed_length_chunking(text, chunk_size=500, overlap=50)
    for chunk in chunks:
        all_chunks.append({
            'text': chunk,
            'title': result['title'],
            'url': result['url']
        })
 
print(f"Total chunks: {len(all_chunks)}")
 
chunk_texts = [c['text'] for c in all_chunks]
chunk_embeddings = get_embeddings(chunk_texts)

def vector_search(query, chunks, chunk_embeddings, top_k=3):
    """Find the most similar chunks to the query."""
    query_embedding = get_embeddings(query)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'similarity': similarities[idx],
            'title': all_chunks[idx]['title'],
            'url': all_chunks[idx]['url']
        })
    return results

query = "quantum computing"
results = vector_search(query, chunk_texts, chunk_embeddings, top_k=3)
 
print(f"Query: '{query}'\n")
print("=" * 60)
for i, r in enumerate(results, 1):
    print(f"\n[{i}] Similarity: {r['similarity']:.3f}")
    print(f"Title: {r['title']}")
    print(f"URL: {r['url']}")
    print(f"Chunk: {r['chunk'][:300]}...")

total_tokens = 17000
import tiktoken
 
# token savings effect

enc = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4, GPT-4-turbo, etc.
# Alternative: enc = tiktoken.encoding_for_model("gpt-4")  # If you want to use a specific model
top_chunks = [r['chunk'] for r in results]
selected_text = "\n\n".join(top_chunks)
selected_tokens = len(enc.encode(selected_text))
 
print(f"Total tokens: {total_tokens}")
print(f"Selected tokens: {selected_tokens}")
print(f"Savings rate: {(1 - selected_tokens/total_tokens)*100:.1f}%")
