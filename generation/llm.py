import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Choose backend: "gemini", "ollama", or "nvidia"
BACKEND = "nvidia"

PROMPT_TEMPLATE = """You are EduBot, an academic assistant for our college.
Answer the student's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the college documents I was given."
Keep the answer short and clear. Mention the source document at the end.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved_chunks: list) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def call_gemini(prompt: str) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable first.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def call_ollama(prompt: str, model: str = "llama3") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def call_nvidia(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Set the NVIDIA_API_KEY environment variable first.")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model="z-ai/glm-5.3-flash",
        messages=[
            {"role": "system", "content": "You are EduBot, an academic assistant for our college."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=False
    )

    # ✅ Correct way to access content
    return completion.choices[0].message.content.strip()


def generate_answer(question: str, retrieved_chunks: list) -> str:
    prompt = build_prompt(question, retrieved_chunks)
    if BACKEND == "gemini":
        return call_gemini(prompt)
    elif BACKEND == "ollama":
        return call_ollama(prompt)
    elif BACKEND == "nvidia":
        return call_nvidia(prompt)
    else:
        raise ValueError(f"Unknown BACKEND: {BACKEND}")


if __name__ == "__main__":
    # Quick manual test
    fake_chunks = [{"source": "attendance_policy.pdf", "text": "Minimum attendance required is 75% per semester."}]
    print(generate_answer("What is the minimum attendance required?", fake_chunks))
