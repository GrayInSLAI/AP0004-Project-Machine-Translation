import os
import json
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


def train_bpe_tokenizer(files, vocab_size=8000, save_path="./bpe_vocab"):
    # 1. Initialize
    tokenizer = ByteLevelBPETokenizer()

    print(f"Training BPE Tokenizer on {files}...")
    print(f"Target Vocab Size: {vocab_size}")

    # 2. Train
    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<s>",  # SOS
            "<pad>",  # PAD
            "</s>",  # EOS
            "<unk>",  # UNK
        ],
    )

    # 3. Post-Process (Add special tokens automatically)
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    # 4. Save
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tokenizer.save_model(save_path)
    print(f"Tokenizer saved to {save_path}")


if __name__ == "__main__":

    raw_text_file = "temp_train_corpus.txt"
    with open("./data/train_100k.jsonl", "r", encoding="utf-8") as f, open(
        raw_text_file, "w", encoding="utf-8"
    ) as out:
        for line in f:
            item = json.loads(line)
            out.write(item["zh"] + "\n")
            out.write(item["en"] + "\n")

    train_bpe_tokenizer([raw_text_file], vocab_size=8000)

    os.remove(raw_text_file)
