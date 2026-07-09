"""Local vector store backed by SQLite.

Replaces the old JSON-file approach that loaded all embeddings into RAM.
Embeddings are stored as binary BLOBs (numpy float32 arrays) for compact storage.
Search still uses numpy cosine similarity (same algorithm, data loaded on demand).
"""

import json
import os
import sqlite3
import threading

import numpy as np
from openai import OpenAI
import google.generativeai as genai


class LocalVectorStore:
    def __init__(self, data_dir="./backend/data"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "vector_store.db")
        os.makedirs(data_dir, exist_ok=True)
        self._model = None
        self._lock = threading.Lock()

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        # Auto-migrate from old JSON format if it exists
        self._migrate_from_json()

    def _create_tables(self):
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    video_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    start REAL,
                    end REAL,
                    text TEXT,
                    video_title TEXT,
                    video_url TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER DEFAULT 0,
                    PRIMARY KEY (video_id, chunk_id),
                    FOREIGN KEY (video_id) REFERENCES videos(video_id)
                )
            """)

    def _migrate_from_json(self):
        """One-time migration from old vector_store.json to SQLite."""
        json_path = os.path.join(self.data_dir, "vector_store.json")
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)

            video_ids = old_data.get("video_ids", [])
            chunks = old_data.get("chunks", [])

            if not chunks:
                # Empty old store, just rename
                os.rename(json_path, json_path + ".bak")
                return

            print(f"Migrating {len(chunks)} chunks from JSON to SQLite...")

            with self._conn:
                for vid in video_ids:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO videos (video_id) VALUES (?)", (vid,)
                    )

                for chunk in chunks:
                    embedding = chunk.get("embedding")
                    emb_blob = None
                    emb_dim = 0
                    if embedding is not None:
                        emb_array = np.array(embedding, dtype=np.float32)
                        emb_blob = emb_array.tobytes()
                        emb_dim = len(embedding)

                    self._conn.execute(
                        """INSERT OR REPLACE INTO chunks
                           (video_id, chunk_id, start, end, text, video_title, video_url, embedding, embedding_dim)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk["video_id"],
                            chunk["chunk_id"],
                            chunk["start"],
                            chunk["end"],
                            chunk["text"],
                            chunk["video_title"],
                            chunk["video_url"],
                            emb_blob,
                            emb_dim,
                        ),
                    )

            # Rename old file as backup
            os.rename(json_path, json_path + ".bak")
            print(f"Migration complete. Old file renamed to {json_path}.bak")

        except Exception as e:
            print(f"Warning: JSON→SQLite migration failed: {e}")

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

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
                task_type="retrieval_document",
            )
            return result["embedding"]

        elif provider == "openai":
            active_key = api_key or os.getenv("OPENAI_API_KEY")
            if not active_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            client = OpenAI(api_key=active_key)
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small",
            )
            return response.data[0].embedding

        elif provider == "openrouter":
            active_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not active_key:
                raise ValueError("OPENROUTER_API_KEY is not set.")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=active_key,
            )
            response = client.embeddings.create(
                input=[text],
                model="openai/text-embedding-3-small",
            )
            return response.data[0].embedding


        elif provider == "local-ai":
            if self._model is None:
                import logging
                logging.getLogger("huggingface_hub.utils._headers").setLevel(logging.ERROR)
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise ImportError(
                        "The 'sentence-transformers' library is not installed. "
                        "Please run `pip install sentence-transformers` to use Local AI embedding."
                    )
                local_model_path = os.path.join(self.data_dir, "models", "multilingual-e5-small")
                if os.path.exists(local_model_path) and os.listdir(local_model_path):
                    print(f"Loading local SentenceTransformer model from {local_model_path}...")
                    self._model = SentenceTransformer(local_model_path)
                else:
                    print("Loading SentenceTransformer model from Hugging Face Hub cache/remote...")
                    self._model = SentenceTransformer("intfloat/multilingual-e5-small")

            # E5 models expect a prefix
            prefixed_text = text
            if not text.startswith("query:") and not text.startswith("passage:"):
                prefix = "query: " if is_query else "passage: "
                prefixed_text = prefix + text

            emb = self._model.encode(prefixed_text, convert_to_numpy=True)
            return emb.tolist()

        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add_video_chunks(self, video_id, chunks, provider="gemini", api_key=None):
        """Generates embeddings for chunks of a video and saves them to the store."""
        with self._lock:
            # Check if video already exists — remove old chunks
            existing = self._conn.execute(
                "SELECT 1 FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
            if existing:
                print(f"Video {video_id} already exists in vector store. Removing old chunks first...")
                self._conn.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
            else:
                self._conn.execute("INSERT INTO videos (video_id) VALUES (?)", (video_id,))

        print(f"Generating embeddings for {len(chunks)} chunks using {provider}...")
        for i, chunk in enumerate(chunks):
            text_to_embed = chunk["text"]
            try:
                if provider == "local":
                    embedding = None
                    emb_blob = None
                    emb_dim = 0
                else:
                    embedding = self.get_embedding(text_to_embed, provider=provider, api_key=api_key, is_query=False)
                    emb_array = np.array(embedding, dtype=np.float32)
                    emb_blob = emb_array.tobytes()
                    emb_dim = len(embedding)

                with self._lock:
                    self._conn.execute(
                        """INSERT OR REPLACE INTO chunks
                           (video_id, chunk_id, start, end, text, video_title, video_url, embedding, embedding_dim)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk["video_id"],
                            chunk["chunk_id"],
                            chunk["start"],
                            chunk["end"],
                            chunk["text"],
                            chunk["video_title"],
                            chunk["video_url"],
                            emb_blob,
                            emb_dim,
                        ),
                    )
            except Exception as e:
                print(f"Error embedding chunk {i + 1}/{len(chunks)}: {e}")
                raise e

        with self._lock:
            self._conn.commit()
        print(f"Successfully added {len(chunks)} chunks for video {video_id}.")

    def get_all_chunks(self):
        """Returns all chunks as a list of dicts (without embeddings, for listing)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, video_id, start, end, text, video_title, video_url FROM chunks"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_stored_embedding_dim(self):
        """Returns the embedding dimension of the first stored chunk, or None if empty.

        Returns 0 if embeddings are None (local provider with no vectors).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding_dim FROM chunks LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return row["embedding_dim"]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------



    def search_local_bm25(self, query, top_k=5, k1=1.5, b=0.75):
        """Performs BM25 similarity search locally on the CPU using numpy."""
        import re as re_mod

        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, video_id, start, end, text, video_title, video_url FROM chunks"
            ).fetchall()
        chunks = [dict(r) for r in rows]

        if not chunks:
            return []

        def tokenize(text):
            return re_mod.findall(r'\b\w+\b', text.lower())

        chunk_docs = []
        doc_lengths = []
        for chunk in chunks:
            words = tokenize(chunk.get("text", ""))
            chunk_docs.append(words)
            doc_lengths.append(len(words))

        num_docs = len(chunks)
        avg_doc_len = np.mean(doc_lengths) if doc_lengths else 1.0
        if avg_doc_len == 0:
            avg_doc_len = 1.0

        query_words = tokenize(query)
        if not query_words or num_docs == 0:
            return [
                {
                    "chunk_id": c["chunk_id"], "start": c["start"], "end": c["end"],
                    "text": c["text"], "video_id": c["video_id"],
                    "video_title": c["video_title"], "video_url": c["video_url"],
                    "score": 0.0,
                }
                for c in chunks[:top_k]
            ]

        # Calculate Document Frequency (DF) only for query terms
        unique_query_words = set(query_words)
        df = {}
        for doc in chunk_docs:
            doc_set = set(doc)
            for word in unique_query_words:
                if word in doc_set:
                    df[word] = df.get(word, 0) + 1

        # Calculate IDF for query terms
        idf = {}
        for word in unique_query_words:
            df_w = df.get(word, 0)
            idf[word] = max(1e-5, np.log(1.0 + (num_docs - df_w + 0.5) / (df_w + 0.5)))

        # Calculate scores
        scores = np.zeros(num_docs)
        for doc_idx, doc in enumerate(chunk_docs):
            if not doc:
                continue
            doc_len = doc_lengths[doc_idx]
            word_counts = {}
            for word in doc:
                if word in unique_query_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            doc_score = 0.0
            for word in unique_query_words:
                tf = word_counts.get(word, 0)
                if tf > 0:
                    idf_w = idf[word]
                    numerator = tf * (k1 + 1.0)
                    denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
                    doc_score += idf_w * (numerator / denominator)
            scores[doc_idx] = doc_score

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "chunk_id": chunks[idx]["chunk_id"],
                "start": chunks[idx]["start"],
                "end": chunks[idx]["end"],
                "text": chunks[idx]["text"],
                "video_id": chunks[idx]["video_id"],
                "video_title": chunks[idx]["video_title"],
                "video_url": chunks[idx]["video_url"],
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]

    def reciprocal_rank_fusion(self, bm25_results, vector_results, k=60):
        """Combines BM25 and Vector Search results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        
        for rank, item in enumerate(bm25_results, start=1):
            key = (item["video_id"], item["chunk_id"])
            if key not in rrf_scores:
                rrf_scores[key] = {"item": item, "score": 0.0}
            rrf_scores[key]["score"] += 1.0 / (k + rank)
            
        for rank, item in enumerate(vector_results, start=1):
            key = (item["video_id"], item["chunk_id"])
            if key not in rrf_scores:
                rrf_scores[key] = {"item": item, "score": 0.0}
            rrf_scores[key]["score"] += 1.0 / (k + rank)
            
        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        
        fused_results = []
        for entry in sorted_items:
            item_copy = dict(entry["item"])
            item_copy["score"] = entry["score"]
            fused_results.append(item_copy)
            
        return fused_results

    def search(self, query, top_k=5, provider="gemini", api_key=None):
        """Performs hybrid BM25 + Vector search using Reciprocal Rank Fusion."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, video_id, start, end, text, video_title, video_url, embedding, embedding_dim FROM chunks"
            ).fetchall()

        if not rows:
            return []

        # Fall back to BM25 search if no embeddings or local provider
        has_missing = any(r["embedding"] is None for r in rows)
        if provider == "local" or has_missing:
            return self.search_local_bm25(query, top_k=top_k)

        # Generate query embedding
        query_emb = self.get_embedding(query, provider=provider, api_key=api_key, is_query=True)
        query_vector = np.array(query_emb)

        # Decode stored embeddings from BLOB, filtering by matching dimension
        query_dim = len(query_vector)
        embeddings = []
        chunks_data = []
        for row in rows:
            if row["embedding"] is not None:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                if len(emb) == query_dim:
                    embeddings.append(emb)
                    chunks_data.append(dict(row))

        if not embeddings:
            return self.search_local_bm25(query, top_k=top_k)

        embeddings_matrix = np.array(embeddings)

        # Cosine similarity for Vector Search
        query_norm = np.linalg.norm(query_vector)
        matrix_norms = np.linalg.norm(embeddings_matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1e-10
        if query_norm == 0:
            query_norm = 1e-10

        similarities = np.dot(embeddings_matrix, query_vector) / (matrix_norms * query_norm)
        
        # Retrieve candidate pool of size max(top_k * 4, 20) for both search types
        pool_k = max(top_k * 4, 20)
        
        # 1. Vector candidates
        vector_top_indices = np.argsort(similarities)[::-1][:pool_k]
        vector_results = [
            {
                "chunk_id": chunks_data[idx]["chunk_id"],
                "start": chunks_data[idx]["start"],
                "end": chunks_data[idx]["end"],
                "text": chunks_data[idx]["text"],
                "video_id": chunks_data[idx]["video_id"],
                "video_title": chunks_data[idx]["video_title"],
                "video_url": chunks_data[idx]["video_url"],
                "score": float(similarities[idx]),
            }
            for idx in vector_top_indices
        ]

        # 2. BM25 candidates
        bm25_results = self.search_local_bm25(query, top_k=pool_k)

        # 3. Fuse results using Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(bm25_results, vector_results)

        # Return top_k
        return fused_results[:top_k]

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def remove_video(self, video_id):
        """Removes a video and all its chunks from the database."""
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
            self._conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
            self._conn.commit()

    def clear_all(self):
        """Resets the vector database entirely."""
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM videos")
            self._conn.commit()
