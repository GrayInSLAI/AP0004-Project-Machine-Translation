import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import json
import sacrebleu
from tqdm import tqdm
from nltk.tokenize.treebank import (
    TreebankWordDetokenizer,
)

from vocab import (
    Vocabulary,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    load_pretrained_embeddings,
)
from dataset import get_dataloader
from models import Encoder, Decoder, Attention, Seq2Seq


# ==========================================
# 1. Argument Parsing
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Train Chinese-to-English Translator")

    # Model Architecture
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="lstm",
        choices=["gru", "lstm"],
        help="Type of RNN: gru or lstm",
    )
    parser.add_argument(
        "--attn_method",
        type=str,
        default="dot",
        choices=["dot", "general", "additive"],
        help="Attention mechanism method",
    )

    # Dimensions & Layers
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument(
        "--hid_dim", type=int, default=512, help="Hidden layer dimension"
    )
    parser.add_argument("--n_layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    # Training Strategy
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument(
        "--tf_ratio", type=float, default=1, help="Teacher Forcing ratio"
    )
    parser.add_argument(
        "--clip", type=float, default=1.0, help="Gradient clipping value"
    )

    # Decoding Strategy
    parser.add_argument(
        "--decode_method",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="Decoding strategy for validation: greedy or beam",
    )
    parser.add_argument(
        "--beam_size", type=int, default=5, help="Beam size for beam search"
    )

    # Paths
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--vocab_dir", type=str, default="./vocab", help="Vocab directory"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="Directory to save models"
    )

    # Embeddings
    parser.add_argument(
        "--zh_embed_path",
        type=str,
        default="./embeddings/cc.zh.300.vec",
        help="Path to pretrained Chinese embeddings",
    )
    parser.add_argument(
        "--en_embed_path",
        type=str,
        default="./embeddings/cc.en.300.vec",
        help="Path to pretrained English embeddings",
    )

    return parser.parse_args()


# ==========================================
# 2. Utils: Initialization
# ==========================================
def init_weights(m):
    """Xavier initialization."""
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
def greedy_decode(model, src, src_len, max_len, device):
    """Batch greedy decoding."""
    model.eval()
    batch_size = src.shape[0]

    outputs = torch.zeros(batch_size, max_len).long().to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src, src_len)
        mask = src != 0

        input_token = torch.tensor([SOS_IDX] * batch_size).to(device)
        outputs[:, 0] = SOS_IDX

        for t in range(1, max_len):
            output, hidden, _ = model.decoder(
                input_token, hidden, encoder_outputs, mask
            )
            top1 = output.argmax(1)
            outputs[:, t] = top1
            input_token = top1

    return outputs


def beam_search_decode_sentence(
    model, src_tensor, src_len_tensor, beam_size, max_len, device
):
    """
    Beam Search for a SINGLE sentence.

    """
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len_tensor)
        mask = src_tensor != 0

        # Beam Node: (score, input_token, hidden_state, sequence_list)
        beam = [(0.0, SOS_IDX, hidden, [SOS_IDX])]

        for _ in range(max_len):
            candidates = []
            all_eos = True

            for score, input_idx, curr_hidden, seq in beam:
                if input_idx == EOS_IDX:
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
        output = model(src, src_len, trg, teacher_forcing_ratio=tf_ratio)

        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, en_vocab, args):
    model.eval()
    epoch_loss = 0
    refs = []
    hyps = []

    detokenizer = TreebankWordDetokenizer()

    with torch.no_grad():
        for _, (src, trg, src_len, _, raw_en_batch) in enumerate(
            tqdm(iterator, desc=f"Evaluating ({args.decode_method})", mininterval=10)
        ):
            src, trg = src.to(model.device), trg.to(model.device)
            src_len = src_len.to(model.device)

            # Loss
            output = model(src, src_len, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            loss = criterion(
                output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1)
            )
            epoch_loss += loss.item()

            # Decoding
            if args.decode_method == "greedy":
                pred_indices = greedy_decode(
                    model, src, src_len, max_len=50, device=model.device
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
                        50,
                        model.device,
                    )
                    pred_indices.append(seq)

            # Processing Text

            # A. References: Use RAW text directly (No detokenization needed)
            for raw_text in raw_en_batch:
                refs.append([raw_text])

            # B. Hypotheses: Detokenize model output
            for seq in pred_indices:
                tokens = []
                for idx in seq:
                    if idx == EOS_IDX:
                        break
                    if idx not in [SOS_IDX, PAD_IDX]:
                        tokens.append(en_vocab.itos(idx))

                hyp_text = detokenizer.detokenize(tokens)
                hyps.append(hyp_text)

    # Calculate BLEU-13a
    bleu_score = sacrebleu.corpus_bleu(hyps, list(zip(*refs)), tokenize="13a")

    return epoch_loss / len(iterator), bleu_score.score


# ==========================================
# 5. Main Function
# ==========================================
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(vars(args), indent=2)}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 1. Load Vocabs
    print("Loading vocabularies...")
    try:
        zh_vocab = Vocabulary.load(os.path.join(args.vocab_dir, "vocab_zh.json"))
        en_vocab = Vocabulary.load(os.path.join(args.vocab_dir, "vocab_en.json"))
    except FileNotFoundError:
        print("Error: Vocab files not found. Run vocab.py first.")
        return

    # 2. Load Datasets
    print("Loading datasets...")
    train_loader = get_dataloader(
        os.path.join(args.data_dir, "train_100k.jsonl"),
        zh_vocab,
        en_vocab,
        args.batch_size,
        shuffle=True,
    )
    valid_loader = get_dataloader(
        os.path.join(args.data_dir, "valid.jsonl"),
        zh_vocab,
        en_vocab,
        args.batch_size,
        shuffle=False,
    )
    test_loader = get_dataloader(
        os.path.join(args.data_dir, "test.jsonl"),
        zh_vocab,
        en_vocab,
        args.batch_size,
        shuffle=False,
    )

    # 3. Build Model
    print("Building model...")
    input_dim = len(zh_vocab)
    output_dim = len(en_vocab)

    attn = Attention(args.hid_dim, args.hid_dim, method=args.attn_method)
    enc = Encoder(
        input_dim,
        args.emb_dim,
        args.hid_dim,
        args.n_layers,
        args.dropout,
        args.rnn_type,
    )
    dec = Decoder(
        output_dim,
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

    # 4. Load Pretrained Embeddings
    # Load ZH
    if os.path.exists(args.zh_embed_path):
        print(f"Loading ZH Embeddings from {args.zh_embed_path}...")
        zh_emb = load_pretrained_embeddings(
            zh_vocab, args.zh_embed_path, embed_dim=args.emb_dim
        )
        model.encoder.embedding.weight.data.copy_(torch.from_numpy(zh_emb))
    else:
        print(f"Warning: ZH embeddings not found at {args.zh_embed_path}")

    # Load EN
    if os.path.exists(args.en_embed_path):
        print(f"Loading EN Embeddings from {args.en_embed_path}...")
        en_emb = load_pretrained_embeddings(
            en_vocab, args.en_embed_path, embed_dim=args.emb_dim
        )
        model.decoder.embedding.weight.data.copy_(torch.from_numpy(en_emb))
    else:
        print(f"Warning: EN embeddings not found at {args.en_embed_path}")

    # 5. Training Loop
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    best_bleu = -1.0

    print("\nStart Training...")
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, args.clip, args.tf_ratio
        )
        valid_loss, valid_bleu = evaluate(
            model, valid_loader, criterion, en_vocab, args
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
                    f"best_model_{args.rnn_type}_{args.attn_method}_{args.tf_ratio}_{args.decode_method}.pt",
                ),
            )
            print(f"\t--> Saved Best Model")

    # 6. Test
    print("\nRunning Test on Best Model...")
    checkpoint = torch.load(
        os.path.join(
            args.save_dir,
            f"best_model_{args.rnn_type}_{args.attn_method}_{args.tf_ratio}_{args.decode_method}.pt",
        )
    )
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_bleu = evaluate(model, test_loader, criterion, en_vocab, args)
    print(f"Test Loss: {test_loss:.3f} | Test BLEU: {test_bleu:.2f}")


if __name__ == "__main__":
    main()
