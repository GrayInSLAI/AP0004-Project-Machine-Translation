import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import data_utils
from vocab import Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX
from tqdm import tqdm


class TranslationDataset(Dataset):
    """
    Optimized Dataset: Pre-tokenizes data and filters out long sequences.
    """

    def __init__(self, file_path, zh_vocab, en_vocab, max_length=128):
        """
        Args:
            max_length: Filter out samples longer than this (tokens).
        """
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab
        self.max_length = max_length
        self.samples = []

        print(f"Loading and Pre-processing data from {file_path}...")

        # Generator yields raw dicts
        raw_generator = data_utils.load_jsonl_generator(file_path)

        kept = 0
        ignored = 0

        for item in tqdm(raw_generator, desc="Pre-processing", mininterval=10):
            raw_zh = item["zh"]
            raw_en = item["en"]

            # 1. Tokenize immediately
            zh_tokens = data_utils.tokenize(raw_zh, "zh")
            en_tokens = data_utils.tokenize(raw_en, "en")

            # 2. Filtering Logic (Not Truncation!)
            # Check length including <sos> and <eos> (+2)
            if len(zh_tokens) + 2 > max_length or len(en_tokens) + 2 > max_length:
                ignored += 1
                continue

            # 3. Numericalize
            src_indices = [SOS_IDX] + zh_vocab.lookup_indices(zh_tokens) + [EOS_IDX]
            trg_indices = [SOS_IDX] + en_vocab.lookup_indices(en_tokens) + [EOS_IDX]

            # 4. Store as Tensors (Ready for GPU)
            # Store tuple: (src_tensor, trg_tensor, raw_zh, raw_en)
            self.samples.append(
                (
                    torch.tensor(src_indices, dtype=torch.long),
                    torch.tensor(trg_indices, dtype=torch.long),
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


def collate_fn(batch):
    """
    Collate function handles padding for variable length tensors.
    """
    src_batch, trg_batch = [], []
    raw_zh_batch, raw_en_batch = [], []

    for src_sample, trg_sample, raw_zh, raw_en in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
        raw_zh_batch.append(raw_zh)
        raw_en_batch.append(raw_en)

    # Pad sequences
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)

    # Calculate lengths
    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)

    return src_padded, trg_padded, src_lengths, raw_zh_batch, raw_en_batch


def get_dataloader(
    file_path, zh_vocab, en_vocab, batch_size=256, shuffle=True, num_workers=0
):
    # Set max_length based on your previous analysis
    dataset = TranslationDataset(file_path, zh_vocab, en_vocab, max_length=128)

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
    print("=== Testing Optimized dataset.py ===")

    try:
        zh_vocab = Vocabulary.load("./vocab/vocab_zh.json")
        en_vocab = Vocabulary.load("./vocab/vocab_en.json")
    except FileNotFoundError:
        print("Run vocab.py first.")
        exit()

    # Test loading
    loader = get_dataloader(
        "./data/train_100k.jsonl", zh_vocab, en_vocab, batch_size=4, num_workers=0
    )

    src, trg, lens, raw_zh, raw_en = next(iter(loader))
    print(f"Batch Shape: {src.shape}")
    print(f"Filtered logic working correctly if you see 'Filtered' count above.")
