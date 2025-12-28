import json
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
import data_utils

# ==========================================
# Constants
# ==========================================
PAD_TOKEN = "<pad>"  # Padding
SOS_TOKEN = "<sos>"  # Start of Sentence
EOS_TOKEN = "<eos>"  # End of Sentence
UNK_TOKEN = "<unk>"  # Unknown Word

# Default indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Vocabulary:
    """
    Vocabulary class to handle token-to-id and id-to-token mapping.
    """

    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.is_built = False

    def add_token(self, token):
        """Add a token to the frequency counter"""
        self.word_count[token] += 1

    def build(self, min_freq=1, max_size=None):
        """
        Build the actual vocabulary based on frequency and size limits.
        """
        # 1. Initialize with specials
        self.word2idx = {k: v for v, k in enumerate(self.specials)}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        current_idx = len(self.specials)

        # 2. Sort words by frequency (High -> Low)
        sorted_words = self.word_count.most_common()

        # Statistics
        unique_tokens = len(self.word_count)
        kept_tokens = 0

        # 3. Add words to vocab
        for word, count in sorted_words:
            # Filter: Frequency
            if count < min_freq:
                continue

            # Filter: Max Size
            if max_size is not None and len(self.word2idx) >= max_size:
                break

            self.word2idx[word] = current_idx
            self.idx2word[current_idx] = word
            current_idx += 1
            kept_tokens += 1

        self.is_built = True

        # Print Statistics
        print(f"[{self.name}] Vocabulary Built.")
        print(f"   Original Unique Tokens: {unique_tokens}")
        print(f"   Final Vocab Size:       {len(self.word2idx)}")
        print(
            f"   Kept Tokens:            {kept_tokens} (Coverage: {kept_tokens/unique_tokens:.2%})"
        )
        print(
            f"   Cutoff (min_freq={min_freq}): {unique_tokens - kept_tokens} tokens discarded."
        )

    def __len__(self):
        return len(self.word2idx)

    def stoi(self, token):
        """Token to Index (returns <unk> if not found)"""
        if not self.is_built:
            raise RuntimeError("Vocabulary not built yet!")
        return self.word2idx.get(token, self.word2idx[UNK_TOKEN])

    def itos(self, idx):
        """Index to Token (returns <unk> if not found)"""
        if not self.is_built:
            raise RuntimeError("Vocabulary not built yet!")
        return self.idx2word.get(idx, UNK_TOKEN)

    def lookup_indices(self, tokens):
        """Convert list of tokens to list of indices"""
        return [self.stoi(token) for token in tokens]

    def lookup_tokens(self, indices):
        """Convert list of indices to list of tokens"""
        return [self.itos(idx) for idx in indices]

    def save(self, path):
        """Save vocabulary to a JSON file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {"name": self.name, "word2idx": self.word2idx}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[{self.name}] Vocab saved to {path}")

    @classmethod
    def load(cls, path):
        """Load vocabulary from a JSON file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocab file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(data["name"])
        vocab.word2idx = data["word2idx"]
        # Rebuild idx2word
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab.is_built = True
        print(f"[{vocab.name}] Vocab loaded. Size: {len(vocab)}")
        return vocab


# ==========================================
# Helper: Pretrained Embeddings
# ==========================================
def load_pretrained_embeddings(vocab, vector_path, embed_dim=300):
    """
    Load pretrained vectors (FastText) and align with vocab.

    Args:
        vocab: The Vocabulary object
        vector_path: Path to .vec file
        embed_dim: Dimension of embeddings

    Returns:
        embedding_matrix: (vocab_size, embed_dim) numpy array
    """
    print(f"Loading pretrained embeddings from {vector_path}...")

    # Initialize matrix with random normal distribution
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))

    hit_count = 0

    with open(vector_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Matching Vectors", mininterval=10):
            parts = line.rstrip().split(" ")
            word = parts[0]

            # If word is in our vocab, update the matrix
            if word in vocab.word2idx:
                idx = vocab.word2idx[word]
                # Some vectors might have header lines or wrong dims
                if len(parts) == embed_dim + 1:
                    vector = np.array(parts[1:], dtype=np.float32)
                    embedding_matrix[idx] = vector
                    hit_count += 1

    # Force special tokens to specific values if needed (optional)
    if PAD_TOKEN in vocab.word2idx:
        embedding_matrix[vocab.word2idx[PAD_TOKEN]] = np.zeros(embed_dim)

    print(f"Pretrained embeddings loaded.")
    print(f"   Vocab Size: {vocab_size}")
    print(f"   Hits:       {hit_count} ({hit_count/vocab_size:.2%})")

    return embedding_matrix


# ==========================================
# Main Execution Logic
# ==========================================
def build_vocab_from_dataset(data_path, min_freq=3, max_size=None):
    """
    Main pipeline to build both Chinese and English vocabs.
    """
    print(f"Scanning {data_path} to build vocabulary...")

    zh_vocab = Vocabulary("Chinese")
    en_vocab = Vocabulary("English")

    generator = data_utils.load_jsonl_generator(data_path)

    for item in tqdm(generator, desc="Counting tokens", total=100000, mininterval=10):
        zh_tokens = data_utils.tokenize(item["zh"], "zh")
        en_tokens = data_utils.tokenize(item["en"], "en")

        for t in zh_tokens:
            zh_vocab.add_token(t)
        for t in en_tokens:
            en_vocab.add_token(t)

    print("-" * 40)
    zh_vocab.build(min_freq=min_freq, max_size=max_size)
    print("-" * 20)
    en_vocab.build(min_freq=min_freq, max_size=max_size)
    print("-" * 40)

    return zh_vocab, en_vocab


if __name__ == "__main__":

    DATA_FILE = "./data/train_100k.jsonl"
    MIN_FREQ = 5
    MAX_SIZE = 25000

    # 1. Build
    zh_v, en_v = build_vocab_from_dataset(
        DATA_FILE, min_freq=MIN_FREQ, max_size=MAX_SIZE
    )

    # 2. Save
    zh_v.save("./vocab/vocab_zh.json")
    en_v.save("./vocab/vocab_en.json")

    # 3. Test Loading Vectors
    fasttext_en_path = "./embeddings/cc.en.300.vec"
    if os.path.exists(fasttext_en_path):
        emb = load_pretrained_embeddings(en_v, fasttext_en_path, embed_dim=300)
        print(f"Embedding matrix shape: {emb.shape}")
