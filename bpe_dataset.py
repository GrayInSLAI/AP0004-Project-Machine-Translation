import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import os
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


class BPEDataset(Dataset):
    """
    Dataset for Byte-Level BPE.
    Loads data, tokenizes it into memory, and filters by length.
    """

    def __init__(self, file_path, tokenizer_path, max_length=128):
        self.max_length = max_length
        self.samples = []

        # 1. Load Tokenizer
        vocab_file = os.path.join(tokenizer_path, "vocab.json")
        merges_file = os.path.join(tokenizer_path, "merges.txt")

        if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
            raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_path}")

        self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)

        # 2. Get Special IDs
        # Assuming you trained with special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
        # Typically: <s>=0, <pad>=1, </s>=2, <unk>=3
        self.sos_id = self.tokenizer.token_to_id("<s>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("</s>")

        # Sanity check
        if self.pad_id is None:
            print("Warning: <pad> token not found. Defaulting pad_id to 1.")
            self.pad_id = 1

        print(
            f"Tokenizer Loaded. PAD ID: {self.pad_id}, SOS ID: {self.sos_id}, EOS ID: {self.eos_id}"
        )

        # 3. Configure Post-Processor
        # This automatically adds <s> at start and </s> at end for every encode()
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.eos_id),
            ("<s>", self.sos_id),
        )
        # Enable truncation logic in tokenizer (optional, as we filter manually too)
        self.tokenizer.enable_truncation(max_length=max_length)

        # 4. Load & Pre-process Data
        print(f"Loading and Pre-tokenizing data from {file_path}...")

        kept = 0
        ignored = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Tokenizing", mininterval=10):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_zh = item["zh"]
                raw_en = item["en"]

                # Encode
                enc_zh = self.tokenizer.encode(raw_zh)
                enc_en = self.tokenizer.encode(raw_en)

                # Filter by length (Including special tokens)
                if len(enc_zh.ids) > max_length or len(enc_en.ids) > max_length:
                    ignored += 1
                    continue

                # Convert to Tensors
                self.samples.append(
                    (
                        torch.tensor(enc_zh.ids, dtype=torch.long),
                        torch.tensor(enc_en.ids, dtype=torch.long),
                        raw_zh,
                        raw_en,
                    )
                )
                kept += 1

        print(
            f"Data Loaded. Kept: {kept}, Filtered: {ignored} (Too long > {max_length})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()


class CollateFunctor:
    """
    Callable class to handle padding using a specific pad_id.
    """

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        src_batch, trg_batch = [], []
        raw_zh, raw_en = [], []

        for src, trg, rz, re in batch:
            src_batch.append(src)
            trg_batch.append(trg)
            raw_zh.append(rz)
            raw_en.append(re)

        # Pad sequences
        src_padded = pad_sequence(
            src_batch, padding_value=self.pad_id, batch_first=True
        )
        trg_padded = pad_sequence(
            trg_batch, padding_value=self.pad_id, batch_first=True
        )

        # Lengths (for pack_padded_sequence)
        src_len = torch.tensor([len(s) for s in src_batch], dtype=torch.long)

        return src_padded, trg_padded, src_len, raw_zh, raw_en


def get_dataloader(
    file_path,
    tokenizer_path,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    max_length=128,
):
    """
    Factory function to create DataLoader with BPE support.
    """
    dataset = BPEDataset(file_path, tokenizer_path, max_length=max_length)

    # Initialize collate function with the correct pad_id from dataset
    collate_fn = CollateFunctor(pad_id=dataset.pad_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


# ==========================================
# Unit Test
# ==========================================
if __name__ == "__main__":
    print("=== Testing bpe_dataset.py ===")

    VOCAB_DIR = "./bpe_vocab"
    DATA_FILE = "./data/train_100k.jsonl"

    if not os.path.exists(os.path.join(VOCAB_DIR, "vocab.json")):
        print("Error: vocab.json not found. Please run your bpe_vocab.py first.")
        exit()

    # 1. Test Loader Creation
    loader = get_dataloader(DATA_FILE, VOCAB_DIR, batch_size=4, num_workers=0)

    # 2. Inspect Batch
    print("\n[Inspecting One Batch]")
    src, trg, lens, raw_zh, raw_en = next(iter(loader))

    print(f"Source Shape: {src.shape}")
    print(f"Target Shape: {trg.shape}")
    print(f"Lengths: {lens}")

    # 3. Decode verification
    # We need to load tokenizer manually to decode for test
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer(
        f"{VOCAB_DIR}/vocab.json", f"{VOCAB_DIR}/merges.txt"
    )

    print("\n[Decoding First Sample]")
    ids = src[0].tolist()
    print(f"IDs: {ids}")
    decoded = tokenizer.decode(ids)  # automatically skips special tokens usually
    print(f"Decoded: {decoded}")
    print(f"Raw: {raw_zh[0]}")

    # Check if Padding is working (assuming pad_id=1)
    if 1 in ids:
        print("Padding detected in IDs (Correct).")
