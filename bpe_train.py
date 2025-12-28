import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import json
import sacrebleu
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer

from bpe_dataset import get_dataloader
from bpe_models import Encoder, Decoder, Attention, Seq2Seq


# ==========================================
# 1. Argument Parsing
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Train NMT with BPE & Shared Embeddings (Standard)"
    )

    # Model Architecture
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["gru", "lstm"])
    parser.add_argument(
        "--attn_method", type=str, default="dot", choices=["dot", "general", "additive"]
    )

    # Dimensions
    parser.add_argument("--emb_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument(
        "--hid_dim", type=int, default=512, help="Hidden layer dimension"
    )
    parser.add_argument("--n_layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    # Training Strategy
    # Standard settings for normal GPUs
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument(
        "--tf_ratio", type=float, default=1, help="Teacher Forcing ratio"
    )
    parser.add_argument(
        "--clip", type=float, default=1.0, help="Gradient clipping value"
    )

    # Standard workers
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Decoding Strategy
    parser.add_argument(
        "--decode_method", type=str, default="greedy", choices=["greedy", "beam"]
    )
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")

    # Paths
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--vocab_dir",
        type=str,
        default="./bpe_vocab",
        help="Directory containing vocab.json and merges.txt",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./bpe_checkpoints",
        help="Directory to save models",
    )

    return parser.parse_args()


# ==========================================
# 2. Utils
# ==========================================
def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param.data)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# 3. Decoding Strategies
# ==========================================
def greedy_decode(model, src, src_len, max_len, sos_id, device):
    """Batch greedy decoding."""
    model.eval()
    batch_size = src.shape[0]

    outputs = torch.zeros(batch_size, max_len).long().to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src, src_len)
        mask = src != 0

        input_token = torch.tensor([sos_id] * batch_size).to(device)
        outputs[:, 0] = sos_id

        for t in range(1, max_len):
            output, hidden, _ = model.decoder(
                input_token, hidden, encoder_outputs, mask
            )
            top1 = output.argmax(1)
            outputs[:, t] = top1
            input_token = top1

    return outputs


def beam_search_decode_sentence(
    model, src_tensor, src_len_tensor, beam_size, max_len, sos_id, eos_id, device
):
    """Beam Search for a SINGLE sentence."""
    model.eval()
    with torch.no_grad():
        actual_len = src_len_tensor.item()
        src_tensor = src_tensor[:, :actual_len]

        encoder_outputs, hidden = model.encoder(src_tensor, src_len_tensor)
        mask = src_tensor != 0

        # Beam Node: (score, input_token, hidden_state, sequence_list)
        beam = [(0.0, sos_id, hidden, [sos_id])]

        for _ in range(max_len):
            candidates = []
            all_eos = True

            for score, input_idx, curr_hidden, seq in beam:
                if input_idx == eos_id:
                    candidates.append((score, input_idx, curr_hidden, seq))
                    continue

                all_eos = False
                input_token_tensor = torch.tensor([input_idx]).to(device)

                prediction, new_hidden, _ = model.decoder(
                    input_token_tensor, curr_hidden, encoder_outputs, mask
                )

                log_probs = torch.log_softmax(prediction, dim=1).squeeze(0)
                topk_log_probs, topk_indices = log_probs.topk(beam_size)

                for k in range(len(topk_log_probs)):
                    idx = topk_indices[k].item()
                    prob = topk_log_probs[k].item()
                    new_score = score + prob
                    new_seq = seq + [idx]
                    candidates.append((new_score, idx, new_hidden, new_seq))

            if all_eos:
                break

            ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
            beam = ordered[:beam_size]

        return beam[0][3]


# ==========================================
# 4. Training & Evaluation
# ==========================================
def train_epoch(model, iterator, optimizer, criterion, clip, tf_ratio):
    model.train()
    epoch_loss = 0

    for _, (src, trg, src_len, _, _) in enumerate(
        tqdm(iterator, desc="Training", leave=False, mininterval=10)
    ):
        src, trg = src.to(model.device), trg.to(model.device)
        src_len = src_len.to(model.device)

        optimizer.zero_grad()

        # Standard Forward Pass (No AMP)
        output = model(src, src_len, trg, teacher_forcing_ratio=tf_ratio)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        # Standard Backward Pass (No Scaler)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, tokenizer, args, special_ids):
    model.eval()
    epoch_loss = 0
    refs = []
    hyps = []

    sos_id, eos_id, _ = special_ids

    with torch.no_grad():
        for _, (src, trg, src_len, _, raw_en_batch) in enumerate(
            tqdm(iterator, desc=f"Evaluating ({args.decode_method})", mininterval=10)
        ):
            src, trg = src.to(model.device), trg.to(model.device)
            src_len = src_len.to(model.device)

            # 1. Loss
            output = model(src, src_len, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            loss = criterion(
                output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1)
            )
            epoch_loss += loss.item()

            # 2. Decoding
            if args.decode_method == "greedy":
                pred_indices = greedy_decode(
                    model, src, src_len, args.max_len, sos_id, model.device
                )
                pred_indices = pred_indices.cpu().numpy().tolist()
            elif args.decode_method == "beam":
                pred_indices = []
                for j in range(src.shape[0]):
                    seq = beam_search_decode_sentence(
                        model,
                        src[j].unsqueeze(0),
                        src_len[j].unsqueeze(0),
                        args.beam_size,
                        args.max_len,
                        sos_id,
                        eos_id,
                        model.device,
                    )
                    pred_indices.append(seq)

            # 3. Process Text
            # A. References: Use RAW text
            for raw_text in raw_en_batch:
                refs.append([raw_text])

            # B. Hypotheses: Decode BPE
            for seq in pred_indices:
                decoded_text = tokenizer.decode(seq, skip_special_tokens=True)
                hyps.append(decoded_text)

    # Calculate BLEU-13a
    bleu_score = sacrebleu.corpus_bleu(hyps, list(zip(*refs)), tokenize="13a")

    return epoch_loss / len(iterator), bleu_score.score


# ==========================================
# 5. Main
# ==========================================
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(vars(args), indent=2)}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 1. Load Tokenizer
    print("Loading BPE Tokenizer...")
    vocab_path = os.path.join(args.vocab_dir, "vocab.json")
    merges_path = os.path.join(args.vocab_dir, "merges.txt")

    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} not found. Run bpe_vocab.py first.")
        return

    tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab Size: {vocab_size}")

    # Get Special IDs
    sos_id = tokenizer.token_to_id("<s>")
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("</s>")

    if pad_id is None:
        pad_id = 1
    if sos_id is None:
        sos_id = 0
    if eos_id is None:
        eos_id = 2

    print(f"Special IDs -> PAD: {pad_id}, SOS: {sos_id}, EOS: {eos_id}")
    special_ids = (sos_id, eos_id, pad_id)

    # 2. Datasets
    print("Loading datasets...")
    train_loader = get_dataloader(
        os.path.join(args.data_dir, "train_100k.jsonl"),
        args.vocab_dir,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_length=args.max_len,
    )

    valid_loader = get_dataloader(
        os.path.join(args.data_dir, "valid.jsonl"),
        args.vocab_dir,
        args.batch_size,
        shuffle=False,
        num_workers=4,
        max_length=args.max_len,
    )

    test_loader = get_dataloader(
        os.path.join(args.data_dir, "test.jsonl"),
        args.vocab_dir,
        args.batch_size,
        shuffle=False,
        num_workers=4,
        max_length=args.max_len,
    )

    # 3. Build Model (Shared Embeddings)
    print("Building model...")

    attn = Attention(args.hid_dim, args.hid_dim, method=args.attn_method)

    enc = Encoder(
        vocab_size,
        args.emb_dim,
        args.hid_dim,
        args.n_layers,
        args.dropout,
        args.rnn_type,
    )
    dec = Decoder(
        vocab_size,
        args.emb_dim,
        args.hid_dim,
        args.hid_dim,
        args.n_layers,
        args.dropout,
        attn,
        args.rnn_type,
    )

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    print(f"Model Parameters: {count_parameters(model):,}")
    print("Note: Using Shared Embeddings.")

    # 4. Training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_bleu = -1.0

    print("\nStart Training...")
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, args.clip, args.tf_ratio
        )
        valid_loss, valid_bleu = evaluate(
            model, valid_loader, criterion, tokenizer, args, special_ids
        )

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        print(f"Epoch: {epoch+1:02} | Time: {int(mins)}m {int(secs)}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}")
        print(
            f"\tVal BLEU: {valid_bleu:.2f} | Best BLEU: {max(best_bleu, valid_bleu):.2f}"
        )

        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_bleu": best_bleu,
                    "args": vars(args),
                },
                os.path.join(
                    args.save_dir,
                    f"bpe_best_model_{args.rnn_type}_{args.attn_method}_{args.tf_ratio}_{args.decode_method}.pt",
                ),
            )
            print(f"\t--> Saved Best Model")

    # 5. Test
    print("\nRunning Test on Best Model...")
    checkpoint = torch.load(
        os.path.join(
            args.save_dir,
            f"bpe_best_model_{args.rnn_type}_{args.attn_method}_{args.tf_ratio}_{args.decode_method}.pt",
        )
    )
    model.load_state_dict(checkpoint["state_dict"])

    test_loss, test_bleu = evaluate(
        model, test_loader, criterion, tokenizer, args, special_ids
    )
    print(f"Test Loss: {test_loss:.3f} | Test BLEU: {test_bleu:.2f}")


if __name__ == "__main__":
    main()
