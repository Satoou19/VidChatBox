import os
import re
from openai import OpenAI
import google.generativeai as genai

class RAGChatbot:
    def __init__(self, gemini_api_key=None, openai_api_key=None, groq_api_key=None, deepseek_api_key=None, openrouter_api_key=None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            
        self.openai_client = None
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            
        self.groq_client = None
        if self.groq_api_key:
            self.groq_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.groq_api_key
            )
            
        self.deepseek_client = None
        if self.deepseek_api_key:
            self.deepseek_client = OpenAI(
                base_url="https://api.deepseek.com/v1",
                api_key=self.deepseek_api_key
            )

        self.openrouter_client = None
        if self.openrouter_api_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )

    def format_timestamp(self, seconds):
        """Converts seconds float to HH:MM:SS format."""
        s = int(seconds)
        hours = s // 3600
        minutes = (s % 3600) // 60
        secs = s % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def build_timestamp_url(self, base_url, seconds):
        """Builds a YouTube/Twitch URL with the timestamp parameter (?t=X or &t=X)."""
        sec_int = int(seconds)
        if "youtube.com" in base_url or "youtu.be" in base_url:
            if "?" in base_url:
                return f"{base_url}&t={sec_int}"
            return f"{base_url}?t={sec_int}"
        elif "twitch.tv" in base_url:
            # Twitch uses ?t=XhYmZs format, e.g. ?t=1h20m15s or just ?t=80m15s
            # For simplicity, we can do ?t=XmYs or use standard twitch timestamp formatting
            minutes = sec_int // 60
            secs = sec_int % 60
            return f"{base_url}?t={minutes}m{secs}s"
        return f"{base_url}#t={sec_int}"

    def construct_system_prompt(self, persona_mode="streamer"):
        """Constructs system instructions.
        
        If persona_mode is 'streamer', it instructs the LLM to adopt the persona
        of the streamer (using context as 'my memories/opinions').
        """
        base_prompt = (
            "You are a helpful AI assistant for VidChatBox, an application that queries VID transcripts. "
            "Your answers must be based ONLY on the provided VID transcript segments. "
            "If the provided context does not contain the answer, say honestly that you cannot find this information in the streamer's VIDs. "
            "Never invent or assume facts outside the provided context.\n\n"
            "CRITICAL CITATION RULES:\n"
            "1. You MUST cite your source using a direct markdown link with the format: `[HH:MM:SS](URL_WITH_TIMESTAMP)`.\n"
            "2. Place the citation link directly next to the statement or paragraph it refers to.\n"
            "3. Use ONLY the exact citation URLs provided in the context. DO NOT construct or alter URLs yourself.\n"
            "4. Example: 'The streamer mentioned that he prefers character builds with high crit rate [01:23:45](https://youtube.com/...&t=5025).'\n"
        )
        
        if persona_mode == "streamer":
            persona_prompt = (
                "\nPERSONA INSTRUCTION:\n"
                "Adopt the persona of the streamer. Speak in the first person ('I', 'me', 'my'). "
                "Treat the context segments as your own memory and spoken opinions. "
                "Keep the tone friendly, conversational, and representative of a live stream (e.g., using casual words, gaming references where applicable). "
                "Remember, even in persona, you must strictly stick to the facts in the context and use the exact citation URLs for timestamps!"
            )
            return base_prompt + persona_prompt
            
        return base_prompt

    def chat_stream(self, query, search_results, provider="gemini", model="", persona_mode="streamer"):
        """Streams chatbot responses using Gemini, OpenAI, Groq, DeepSeek, or OpenRouter."""
        # 1. Format the context
        context_str = "--- CONTEXT SEGMENTS FROM VID TRANSCRIPTS ---\n"
        for i, chunk in enumerate(search_results):
            start_fmt = self.format_timestamp(chunk["start"])
            end_fmt = self.format_timestamp(chunk["end"])
            timestamp_url = self.build_timestamp_url(chunk["video_url"], chunk["start"])
            
            context_str += (
                f"Segment #{i+1}:\n"
                f"Video Title: {chunk['video_title']}\n"
                f"Timestamp: {start_fmt} - {end_fmt}\n"
                f"Citation Link URL: {timestamp_url}\n"
                f"Spoken Text: \"{chunk['text']}\"\n"
                f"-----------------------------------------\n"
            )
            
        system_prompt = self.construct_system_prompt(persona_mode)
        
        # 2. Call LLM stream
        if provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
                
            model_name = model if model else "gemini-1.5-flash"
            model_obj = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
            
            prompt_content = f"User Question: {query}\n\n{context_str}"
            response = model_obj.generate_content(prompt_content, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        elif provider == "openai":
            if not self.openai_client:
                raise ValueError("OPENAI_API_KEY is not set.")
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {query}\n\n{context_str}"}
            ]
            
            model_name = model if model else "gpt-4o-mini"
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content
                    
        elif provider == "groq":
            if not self.groq_client:
                raise ValueError("GROQ_API_KEY is not set.")
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {query}\n\n{context_str}"}
            ]
            
            model_name = model if model else "llama-3.1-8b-instant"
            if model_name == "llama-3.3-70b-specdec":
                model_name = "llama-3.3-70b-versatile"
                
            response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content
                    
        elif provider == "deepseek":
            if not self.deepseek_client:
                raise ValueError("DEEPSEEK_API_KEY is not set.")
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {query}\n\n{context_str}"}
            ]
            
            model_name = model if model else "deepseek-chat"
            response = self.deepseek_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content

        elif provider == "openrouter":
            if not self.openrouter_client:
                raise ValueError("OPENROUTER_API_KEY is not set.")
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {query}\n\n{context_str}"}
            ]
            
            model_name = model if model else "google/gemini-2.0-flash"
            response = self.openrouter_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "VidChatBox"
                }
            )
            
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content
        else:
            raise ValueError(f"Unknown chat provider: {provider}")
