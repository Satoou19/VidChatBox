import os
import json
import re

class TranscriptProcessor:
    def __init__(self, data_dir="./backend/data"):
        self.data_dir = data_dir
        self.knowledge_dir = os.path.join(data_dir, "knowledge")
        os.makedirs(self.knowledge_dir, exist_ok=True)

    def clean_text(self, text):
        """Cleans transcript text by stripping extra whitespace, fixing encoding artifacts, etc."""
        if not text:
            return ""
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_transcript(self, segments, chunk_size_words=200, overlap_words=40):
        """Splits transcript segments into overlapping chunks of a target word size.
        
        Maintains start and end timestamps for each chunk.
        """
        chunks = []
        if not segments:
            return chunks

        current_chunk_words = []
        current_chunk_segments = []
        
        segment_idx = 0
        while segment_idx < len(segments):
            seg = segments[segment_idx]
            text = self.clean_text(seg.get("text", ""))
            if not text:
                segment_idx += 1
                continue
                
            seg_words = text.split()
            current_chunk_words.extend(seg_words)
            current_chunk_segments.append(seg)
            
            # Check if we reached the word target or if it's the last segment
            word_count = len(current_chunk_words)
            if word_count >= chunk_size_words or segment_idx == len(segments) - 1:
                # Create chunk
                start_time = current_chunk_segments[0].get("start", 0.0)
                end_time = current_chunk_segments[-1].get("end", 0.0)
                chunk_text = " ".join(current_chunk_words)
                
                chunks.append({
                    "start": start_time,
                    "end": end_time,
                    "text": chunk_text
                })
                
                # Setup next chunk with overlap
                if overlap_words > 0 and word_count > overlap_words and segment_idx < len(segments) - 1:
                    # Slide back: find how many segments from the end we need to keep for overlap
                    overlap_collected = 0
                    overlap_segs = []
                    for rev_seg in reversed(current_chunk_segments):
                        rev_words = len(self.clean_text(rev_seg.get("text", "")).split())
                        overlap_collected += rev_words
                        overlap_segs.append(rev_seg)
                        if overlap_collected >= overlap_words:
                            break
                    
                    # Reverse back to normal order
                    overlap_segs.reverse()
                    
                    # Reset variables for next chunk
                    current_chunk_segments = overlap_segs
                    current_chunk_words = []
                    for o_seg in overlap_segs:
                        current_chunk_words.extend(self.clean_text(o_seg.get("text", "")).split())
                        
                    # Increment index normally
                    segment_idx += 1
                else:
                    # Clear and move to next
                    current_chunk_words = []
                    current_chunk_segments = []
                    segment_idx += 1
            else:
                segment_idx += 1

        # Fallback for remaining words if any
        if current_chunk_words:
            start_time = current_chunk_segments[0].get("start", 0.0)
            end_time = current_chunk_segments[-1].get("end", 0.0)
            chunks.append({
                "start": start_time,
                "end": end_time,
                "text": " ".join(current_chunk_words)
            })

        return chunks

    def save_knowledge_package(self, video_metadata, segments, chunk_size_words=200, overlap_words=40):
        """Processes segments, builds metadata, chunks the text, and exports the JSON package."""
        video_id = video_metadata["id"]
        
        # Clean video_id for file system
        safe_video_id = re.sub(r'[^\w\-_]', '', video_id)
        
        processed_chunks = self.chunk_transcript(segments, chunk_size_words, overlap_words)
        
        # Add index and video reference to each chunk
        chunks_with_meta = []
        for idx, chunk in enumerate(processed_chunks):
            chunks_with_meta.append({
                "chunk_id": idx + 1,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "video_id": video_id,
                "video_title": video_metadata["title"],
                "video_url": video_metadata["webpage_url"]
            })
            
        knowledge_package = {
            "metadata": {
                "video_id": video_id,
                "title": video_metadata["title"],
                "duration": video_metadata.get("duration"),
                "uploader": video_metadata.get("uploader"),
                "url": video_metadata["webpage_url"],
                "total_chunks": len(chunks_with_meta)
            },
            "chunks": chunks_with_meta,
            "segments": segments
        }
        
        output_file = os.path.join(self.knowledge_dir, f"video_{safe_video_id}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(knowledge_package, f, indent=2, ensure_ascii=False)
            
        return output_file, chunks_with_meta
