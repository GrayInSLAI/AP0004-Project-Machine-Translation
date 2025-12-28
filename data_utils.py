import json
import os
import unicodedata
import jieba
import nltk


# ==========================================
# 1. Environment Setup
# ==========================================
def configure_nltk():
    """
    Configure NLTK path to use local data and check for 'punkt'.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_path = os.path.join(current_dir, "nltk_data")

    # Add local path if exists
    if os.path.exists(nltk_data_path):
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.append(nltk_data_path)

    # Check availability once at startup to avoid overhead in loops
    try:
        nltk.data.find("tokenizers/punkt")
        return True
    except LookupError:
        print(
            "Warning: NLTK 'punkt' tokenizer not found. 'tokenize_en' will fall back to split()."
        )
        return False


# Initialize configuration immediately and store state
HAS_PUNKT = configure_nltk()


# ==========================================
# 2. Cleaning Tools
# ==========================================
def normalize_string(s):
    """
    Normalize Unicode string (Full-width -> Half-width, etc.)
    and remove non-printable characters.
    """
    if not isinstance(s, str):
        return ""

    # NFKC normalization (e.g., １ -> 1, Ａ -> A)
    s = unicodedata.normalize("NFKC", s)

    # Remove non-printable characters (control chars)
    s = "".join(ch for ch in s if ch.isprintable())

    # Strip whitespace
    return s.strip()


# ==========================================
# 3. Tokenization Tools
# ==========================================
def tokenize_zh(text):
    """
    Tokenize Chinese text using Jieba.
    """
    # 1. Normalize first
    text = normalize_string(text)

    # 2. Cut
    tokens = jieba.lcut(text)

    # 3. Remove pure whitespace tokens
    return [t.strip() for t in tokens if t.strip()]


def tokenize_en(text):
    """
    Tokenize English text using NLTK (if available) or split.
    """
    # 1. Normalize first
    text = normalize_string(text)

    # 2. Lowercase (Standard for baseline)
    text = text.lower()

    # 3. Tokenize
    if HAS_PUNKT:
        return nltk.word_tokenize(text)
    else:
        # Fallback only if global check failed
        return text.split()


def tokenize(text, lang):
    """
    Wrapper function to dispatch tokenization.
    """
    if lang == "zh":
        return tokenize_zh(text)
    elif lang == "en":
        return tokenize_en(text)
    else:
        raise ValueError(f"Unsupported language: {lang}")


# ==========================================
# 4. File Loading Tools
# ==========================================
def load_jsonl_generator(file_path):
    """
    Generator to yield raw lines from JSONL.
    Generates: {'zh': str, 'en': str, ...}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip broken lines
