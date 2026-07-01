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
        self._model = None
        
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

    def get_embedding(self, text, provider="gemini", api_key=None, is_query=False):
        """Generates embedding for a given text using Gemini, OpenAI, or local sentence-transformers."""
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
        elif provider == "local-ai":
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise ImportError(
                        "The 'sentence-transformers' library is not installed. "
                        "Please run `pip install sentence-transformers` to use Local AI embedding."
                    )
                # Load the multilingual-e5-small model
                self._model = SentenceTransformer("intfloat/multilingual-e5-small")
            
            # E5 models expect a prefix: "query: " for queries and "passage: " for passages
            prefixed_text = text
            if not text.startswith("query:") and not text.startswith("passage:"):
                prefix = "query: " if is_query else "passage: "
                prefixed_text = prefix + text
                
            emb = self._model.encode(prefixed_text, convert_to_numpy=True)
            return emb.tolist()
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
                if provider == "local":
                    embedding = None
                else:
                    embedding = self.get_embedding(text_to_embed, provider=provider, api_key=api_key, is_query=False)
                
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

    def search_local_tfidf(self, query, top_k=5):
        """Performs TF-IDF similarity search locally on the CPU using numpy."""
        import re
        chunks = self.data["chunks"]
        if not chunks:
            return []

        def tokenize(text):
            return re.findall(r'\b\w+\b', text.lower())

        # Build vocabulary and term frequencies for chunks
        chunk_docs = []
        all_words = set()
        for chunk in chunks:
            words = tokenize(chunk.get("text", ""))
            chunk_docs.append(words)
            all_words.update(words)

        vocab = list(all_words)
        vocab_idx = {word: i for i, word in enumerate(vocab)}
        num_docs = len(chunks)

        if num_docs == 0 or not vocab:
            # Fallback if no vocabulary
            return [
                {
                    "chunk_id": chunk["chunk_id"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                    "video_id": chunk["video_id"],
                    "video_title": chunk["video_title"],
                    "video_url": chunk["video_url"],
                    "score": 0.0
                }
                for chunk in chunks[:top_k]
            ]

        # Calculate Document Frequency (DF)
        df = {}
        for doc in chunk_docs:
            seen = set(doc)
            for word in seen:
                df[word] = df.get(word, 0) + 1

        # Calculate IDF
        idf = {}
        for word in vocab:
            idf[word] = np.log(1 + (num_docs / (df[word] + 1)))

        # Calculate TF-IDF matrix
        tfidf_matrix = np.zeros((num_docs, len(vocab)))
        for doc_idx, doc in enumerate(chunk_docs):
            if not doc:
                continue
            word_counts = {}
            for word in doc:
                word_counts[word] = word_counts.get(word, 0) + 1
            for word, count in word_counts.items():
                tf = count / len(doc)
                tfidf_matrix[doc_idx, vocab_idx[word]] = tf * idf[word]

        # Calculate TF-IDF query vector
        query_words = tokenize(query)
        query_tfidf = np.zeros(len(vocab))
        if query_words:
            query_word_counts = {}
            for word in query_words:
                if word in vocab_idx:
                    query_word_counts[word] = query_word_counts.get(word, 0) + 1
            for word, count in query_word_counts.items():
                tf = count / len(query_words)
                query_tfidf[vocab_idx[word]] = tf * idf[word]

        # Compute cosine similarity
        query_norm = np.linalg.norm(query_tfidf)
        matrix_norms = np.linalg.norm(tfidf_matrix, axis=1)

        matrix_norms[matrix_norms == 0] = 1e-10
        if query_norm == 0:
            query_norm = 1e-10

        dot_products = np.dot(tfidf_matrix, query_tfidf)
        similarities = dot_products / (matrix_norms * query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            chunk = chunks[idx]
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

    def search(self, query, top_k=5, provider="gemini", api_key=None):
        """Performs cosine similarity search using numpy."""
        if not self.data["chunks"]:
            return []

        # Fall back to local search if requested or if any chunk is missing its embedding
        has_missing_embeddings = any(chunk.get("embedding") is None for chunk in self.data["chunks"])
        if provider == "local" or has_missing_embeddings:
            return self.search_local_tfidf(query, top_k=top_k)

        # Generate query embedding
        query_emb = self.get_embedding(query, provider=provider, api_key=api_key, is_query=True)
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
