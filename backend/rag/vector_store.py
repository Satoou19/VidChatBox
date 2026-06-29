import os
import json
import numpy as np
from openai import OpenAI
import google.generativeai as genai

class LocalVectorStore:
    def __init__(self, data_dir="./backend/data"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "vector_store.json")
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data if it exists
        self.data = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading vector store JSON, creating new: {e}")
        return {"video_ids": [], "chunks": []}

    def _save_db(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get_embedding(self, text, provider="gemini", api_key=None):
        """Generates embedding for a given text using Gemini or OpenAI."""
        if provider == "gemini":
            active_key = api_key or os.getenv("GEMINI_API_KEY")
            if not active_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            genai.configure(api_key=active_key)
            result = genai.embed_content(
                model="models/gemini-embedding-2",
                content=text,
                task_type="retrieval_document"
            )
            return result["embedding"]
            
        elif provider == "openai":
            active_key = api_key or os.getenv("OPENAI_API_KEY")
            if not active_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            client = OpenAI(api_key=active_key)
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding

        elif provider == "openrouter":
            active_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not active_key:
                raise ValueError("OPENROUTER_API_KEY is not set.")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=active_key
            )
            response = client.embeddings.create(
                input=[text],
                model="openai/text-embedding-3-small"
            )
            return response.data[0].embedding
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def add_video_chunks(self, video_id, chunks, provider="gemini", api_key=None):
        """Generates embeddings for chunks of a video and saves them to the store."""
        # Check if video already exists
        if video_id in self.data["video_ids"]:
            print(f"Video {video_id} already exists in vector store. Removing old chunks first...")
            self.data["chunks"] = [c for c in self.data["chunks"] if c["video_id"] != video_id]
        else:
            self.data["video_ids"].append(video_id)

        print(f"Generating embeddings for {len(chunks)} chunks using {provider}...")
        for i, chunk in enumerate(chunks):
            # Clean chunk text
            text_to_embed = chunk["text"]
            try:
                embedding = self.get_embedding(text_to_embed, provider=provider, api_key=api_key)
                
                # Append to our local store
                chunk_data = {
                    "chunk_id": chunk["chunk_id"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                    "video_id": chunk["video_id"],
                    "video_title": chunk["video_title"],
                    "video_url": chunk["video_url"],
                    "embedding": embedding
                }
                self.data["chunks"].append(chunk_data)
            except Exception as e:
                print(f"Error embedding chunk {i+1}/{len(chunks)}: {e}")
                raise e
        
        self._save_db()
        print(f"Successfully added {len(chunks)} chunks for video {video_id}.")

    def search(self, query, top_k=5, provider="gemini", api_key=None):
        """Performs cosine similarity search using numpy."""
        if not self.data["chunks"]:
            return []

        # Generate query embedding
        query_emb = self.get_embedding(query, provider=provider, api_key=api_key)
        query_vector = np.array(query_emb)
        
        # Build matrices for vector operations
        embeddings = []
        for chunk in self.data["chunks"]:
            embeddings.append(chunk["embedding"])
            
        embeddings_matrix = np.array(embeddings) # shape: (num_chunks, dim)

        if len(query_vector) != embeddings_matrix.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: query vector has dim {len(query_vector)} "
                f"but database chunks have dim {embeddings_matrix.shape[1]}."
            )
            
        query_norm = np.linalg.norm(query_vector)
        results = []
        
        # Compute cosine similarity
        # Cos_sim = dot(A, B) / (norm(A) * norm(B))
        dot_products = np.dot(embeddings_matrix, query_vector)
        matrix_norms = np.linalg.norm(embeddings_matrix, axis=1)
        
        # Avoid division by zero
        matrix_norms[matrix_norms == 0] = 1e-10
        if query_norm == 0:
            query_norm = 1e-10
            
        similarities = dot_products / (matrix_norms * query_norm)
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        for idx in top_indices:
            score = float(similarities[idx])
            chunk = self.data["chunks"][idx]
            
            # Return result without the heavy embedding array to save memory/bandwidth
            result_chunk = {
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "video_id": chunk["video_id"],
                "video_title": chunk["video_title"],
                "video_url": chunk["video_url"],
                "score": score
            }
            results.append(result_chunk)
            
        return results

    def remove_video(self, video_id):
        """Removes a video's ID and all its corresponding chunks from the database."""
        if video_id in self.data["video_ids"]:
            self.data["video_ids"].remove(video_id)
        self.data["chunks"] = [c for c in self.data["chunks"] if c["video_id"] != video_id]
        self._save_db()

    def clear_all(self):
        """Resets the vector database entirely."""
        self.data = {"video_ids": [], "chunks": []}
        self._save_db()
