# CLEAN GENERATOR.PY (FINAL)

from pathlib import Path
import json
import os
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib
import requests

# Config
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIR = Path(os.getenv("EMBED_DIR", "models/embeddings"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
HNSW_INDEX_FILENAME = "hnsw.index"


def _ensure_assets() -> None:
    if not EMBED_DIR.exists():
        raise FileNotFoundError(f"Embeddings dir not found: {EMBED_DIR}")
    for fn in ("embeddings.npy", "meta.json", HNSW_INDEX_FILENAME):
        if not (EMBED_DIR / fn).exists():
            raise FileNotFoundError(f"Missing required file: {EMBED_DIR / fn}")


def retrieve(prompt: str, top_k: int = 3) -> List[Dict[str, Any]]:
    _ensure_assets()

    embeddings = np.load(EMBED_DIR / "embeddings.npy")
    with open(EMBED_DIR / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(EMBED_DIR / HNSW_INDEX_FILENAME))
    index.set_ef(50)

    embedder = SentenceTransformer(EMBED_MODEL)
    q_emb = embedder.encode([prompt], convert_to_numpy=True)

    k = min(top_k, len(meta))
    labels, _ = index.knn_query(q_emb, k=k)

    return [meta[int(i)] for i in labels[0]]


def build_persona_prompt(retrieved, user_prompt):
    examples = "\n\n".join(f"EXAMPLE: {r['text']}" for r in retrieved) if retrieved else "No examples found."

    return f"""
Match this writing style based on examples:

{examples}

User said: {user_prompt}

Reply in that exact tone and style.
""".strip()


def call_ollama(
    prompt: str,
    model_name: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 300,
) -> str:
    """
    Call Ollama with a normal (non-streaming) JSON request
    and return the final text response.
    """
    model = model_name or DEFAULT_OLLAMA_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,  # <- important: no streaming, no iter_lines circus
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=15)
    except requests.exceptions.ConnectionError:
        return (
            "[Ollama Offline] Could not connect to local Ollama daemon at http://localhost:11434.\n"
            "Please start Ollama using `ollama serve` and pull the model (`ollama pull llama3`)."
        )
    except Exception as e:
        return f"Request failed: {e}"

    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        return f"Ollama returned status {resp.status_code}: {err}"

    # Try to parse JSON response
    try:
        data = resp.json()
    except Exception:
        # not JSON, just return raw text
        return resp.text

    # Common direct keys
    for key in ("response", "output", "text", "content"):
        if key in data and isinstance(data[key], str):
            return data[key]

    # OpenAI-style / choices-shaped responses
    if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
        c = data["choices"][0]
        if isinstance(c, dict):
            for k in ("message", "content", "text", "output"):
                if k in c:
                    v = c[k]
                    if isinstance(v, str):
                        return v
                    if isinstance(v, dict):
                        return v.get("content") or v.get("text") or str(v)

    # fallback: dump JSON
    return f"Ollama returned unexpected JSON: {json.dumps(data, ensure_ascii=False)}"


def generate_response(user_prompt, top_k=3, temperature=0.6, model_name=None):
    retrieved = retrieve(user_prompt, top_k=top_k)
    persona_prompt = build_persona_prompt(retrieved, user_prompt)
    return call_ollama(persona_prompt, model_name=model_name, temperature=temperature)


class PersonaGenerator:
    """
    Thin wrapper so the Streamlit app can use:
        - .check_ollama_status()
        - .generate(prompt, temperature)
    while reusing the functional code above.
    """
    def __init__(self, trainer=None):
        self.trainer = trainer

    def check_ollama_status(self) -> bool:
        """Ping Ollama to see if it's alive."""
        try:
            # Derive base URL from OLLAMA_URL (".../api/generate" -> ".../api/tags")
            if "/api/" in OLLAMA_URL:
                base = OLLAMA_URL.split("/api/", 1)[0]
            else:
                base = OLLAMA_URL.rstrip("/")
            status_url = base + "/api/tags"

            resp = requests.get(status_url, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, temperature: float = 0.6) -> str:
        """Generate a response in the user's style."""
        return generate_response(
            user_prompt=prompt,
            top_k=3,
            temperature=temperature,
            model_name=None,  # or set a specific model if you want
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-q", required=True)
    args = parser.parse_args()
    print(generate_response(args.prompt))
