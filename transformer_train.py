import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import argparse
import json
import sacrebleu
from tqdm import tqdm

from bpe_dataset import get_dataloader
from transformer_models import Transformer


# ==========================================
# Utils: Noam Scheduler
# ==========================================
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = (self.d_model**-0.5) * min(
            self.current_step**-0.5, self.current_step * (self.warmup_steps**-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ==========================================
# Utils: Decoding & Eval
# ==========================================
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    batch_size = src.size(0)
    memory = model.encode(src, src_key_padding_mask=src_mask)
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

    for _ in range(max_len - 1):
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device)
        out = model.decode(ys, memory, tgt_mask=tgt_mask)
        prob = model.output_layer(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
        finished |= next_word.squeeze() == end_symbol
        if finished.all():
            break
    return ys


def evaluate_loss_and_bleu(model, loader, criterion, device, pad_id, tokenizer):
    model.eval()
    total_loss = 0
    hypotheses, references = [], []
    sos_id, eos_id = tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>")

    with torch.no_grad():
        for src, tgt, _, _, raw_en in tqdm(
            loader, desc="Evaluating", leave=False, mininterval=10
        ):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            src_mask, tgt_mask = (src == pad_id), (tgt_input == pad_id)

            logits = model(
                src,
                tgt_input,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask,
            )
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
            )
            total_loss += loss.item()

            generated_ids = greedy_decode(
                model, src, src_mask, 50, sos_id, eos_id, device
            )
            preds = tokenizer.decode_batch(
                generated_ids.cpu().tolist(), skip_special_tokens=True
            )
            hypotheses.extend(preds)
            references.extend(raw_en)

    avg_loss = total_loss / len(loader)
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a")
    return avg_loss, bleu_score.score, hypotheses[0], references[0]


# ==========================================
# File Management (Unified)
# ==========================================
def save_everything(output_dir, args, model, optimizer, epoch, best_bleu, history):
    """Saves config, model, and plots to the same directory."""

    # 1. Save Config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # 2. Save Best Model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "best_bleu": best_bleu,
    }
    torch.save(checkpoint, os.path.join(output_dir, "best_model.pt"))

    # 3. Save Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["valid_loss"], label="Valid Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["valid_bleu"], label="Valid BLEU", color="orange")
    plt.title("BLEU Score")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "metrics.png"))
    plt.close()


# ==========================================
# Main Training Logic
# ==========================================
def train_epoch(model, loader, optimizer, scheduler, criterion, device, pad_id):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training", leave=False, mininterval=10)

    for batch_idx, (src, tgt, _, _, _) in enumerate(progress_bar):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
        src_mask, tgt_mask = (src == pad_id), (tgt_input == pad_id)

        optimizer.zero_grad()
        logits = model(
            src, tgt_input, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask
        )
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if batch_idx % 50 == 0:
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_lr():.6f}"}
            )

    return total_loss / len(loader)


def get_args():
    parser = argparse.ArgumentParser()
    # Data & Paths
    parser.add_argument("--train_file", type=str, default="./data/train_100k.jsonl")
    parser.add_argument("--valid_file", type=str, default="./data/valid.jsonl")
    parser.add_argument("--test_file", type=str, default="./data/test.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="./bpe_vocab")

    # === Key Change: Unified Output Directory ===
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments/baseline",
        help="Folder to save all outputs (checkpoints, plots, config)",
    )

    # Model (Optimized Defaults)
    parser.add_argument(
        "--norm_type", type=str, default="rms", choices=["layer", "rms"]
    )  # layer, rms
    parser.add_argument(
        "--pos_scheme",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable"],
    )  # sinusoidal, learnable
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=2048) 
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--batch_size", type=int, default=256)  # 128, 256, 512
    parser.add_argument("--lr", type=float, default=0.0005)  # 0.001, 0.0005, 0.0001
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = get_args()

    # 1. Setup Directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n=== Output Directory: {args.output_dir} ===")
    print(
        f"Config: RMSNorm, Sinusoidal, Layers={args.num_layers}, Dropout={args.dropout}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data
    print("Loading Data...")
    train_loader = get_dataloader(
        args.train_file,
        args.tokenizer_path,
        args.batch_size,
        True,
        max_length=args.max_length,
    )
    valid_loader = get_dataloader(
        args.valid_file,
        args.tokenizer_path,
        args.batch_size,
        False,
        max_length=args.max_length,
    )
    test_loader = get_dataloader(
        args.test_file,
        args.tokenizer_path,
        args.batch_size,
        False,
        max_length=args.max_length,
    )
    tokenizer = train_loader.dataset.tokenizer
    pad_id = train_loader.dataset.pad_id
    vocab_size = tokenizer.get_vocab_size()

    # 3. Model
    model = Transformer(
        vocab_size,
        vocab_size,
        args.d_model,
        args.n_head,
        args.num_layers,
        args.num_layers,
        args.dim_feedforward,
        args.dropout,
        args.norm_type,
        args.pos_scheme,
        args.max_length,
    ).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # 4. Optim
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, args.d_model, args.warmup_steps)
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=args.label_smoothing
    )

    # 5. Loop
    best_bleu = -1.0
    history = {"train_loss": [], "valid_loss": [], "valid_bleu": []}

    print("Starting Training...")
    for epoch in range(args.epochs):
        start_t = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, pad_id
        )
        valid_loss, valid_bleu, sample_hyp, sample_ref = evaluate_loss_and_bleu(
            model, valid_loader, criterion, device, pad_id, tokenizer
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_bleu"].append(valid_bleu)

        print(
            f"Epoch {epoch+1:02} | Time: {time.time()-start_t:.0f}s | T_Loss: {train_loss:.3f} | V_Loss: {valid_loss:.3f} | BLEU: {valid_bleu:.2f}"
        )
        print(f" >> Sample: {sample_hyp}")

        # Save everything to the unified folder if we have a new best
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            print(f" >> New Best! Saving to {args.output_dir}")
            save_everything(
                args.output_dir, args, model, optimizer, epoch + 1, best_bleu, history
            )

    # 6. Final Test
    print("\n=== Final Test Evaluation ===")
    best_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path)["model_state_dict"])
        test_loss, test_bleu, _, _ = evaluate_loss_and_bleu(
            model, test_loader, criterion, device, pad_id, tokenizer
        )
        print(f"Test Set -> Loss: {test_loss:.4f} | BLEU: {test_bleu:.2f}")
    else:
        print("No best model found.")


if __name__ == "__main__":
    main()
