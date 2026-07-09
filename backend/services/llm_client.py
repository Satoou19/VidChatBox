"""LLM client for transcript polishing.

Supports Gemini, OpenAI, Groq, DeepSeek, and OpenRouter providers.
Extracted from the old monolithic main.py.
"""

import os
import re

import google.generativeai as genai
from openai import OpenAI


def call_llm_for_polishing(
    text: str,
    system_prompt: str,
    provider: str,
    model: str,
    gemini_key: str,
    openai_key: str,
    groq_key: str,
    deepseek_key: str,
    openrouter_key: str,
    llm7_key: str,
) -> str:
    """Sends text to the selected LLM provider and returns the polished result.

    Raises:
        ValueError: If the required API key is not set or the provider is unknown.
    """
    if provider == "gemini":
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=gemini_key)
        model_name = model if model else "gemini-3.5-flash"
        model_obj = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )
        response = model_obj.generate_content(text)
        result = response.text.strip()

    elif provider == "openai":
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=openai_key)
        model_name = model if model else "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model_name,  # Fix #4: was incorrectly `model_name=model_name`
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        result = response.choices[0].message.content.strip()

    elif provider == "groq":
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not set.")
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
        )
        model_name = model if model else "llama-3.3-70b-versatile"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        result = response.choices[0].message.content.strip()

    elif provider == "deepseek":
        if not deepseek_key:
            raise ValueError("DEEPSEEK_API_KEY is not set.")
        client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=deepseek_key,
        )
        model_name = model if model else "deepseek-chat"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        result = response.choices[0].message.content.strip()

    elif provider == "openrouter":
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )
        model_name = model if model else "google/gemini-2.0-flash"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "VidChatBox",
            },
        )
        result = response.choices[0].message.content.strip()

    elif provider == "llm7":
        if not llm7_key:
            raise ValueError("LLM7_API_KEY is not set.")
        client = OpenAI(
            base_url="https://api.llm7.io/v1",
            api_key=llm7_key,
        )
        model_name = model if model else "default"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        result = response.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown provider for polishing: {provider}")

    # Remove any markdown wrapping if the LLM outputted them despite instructions
    if result.startswith("```"):
        result = re.sub(r"^```(?:markdown)?\n", "", result)
        result = re.sub(r"\n```$", "", result)
        result = result.strip()

    return result
