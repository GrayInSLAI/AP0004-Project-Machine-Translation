import os
import sys
import json
import numpy as np
import torch
import sacrebleu
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ==============================================================================
# 1. Setup
# ==============================================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_USE_CPU"] = "false"

MODEL_PATH = "./models/mt5-base"
DATA_DIR = "./data"
OUTPUT_DIR = "./mt5_translation_ft"

# ==============================================================================
# 2. Advanced Hyperparameters
# ==============================================================================
BATCH_SIZE = 64
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
PREFIX = "translate Chinese to English: "


def run_training():
    print(f"---  Starting PRO Training on {torch.cuda.get_device_name(0)} ---")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    # Loda datasets
    data_files = {
        "train": os.path.join(DATA_DIR, "train_100k.jsonl"),
        "validation": os.path.join(DATA_DIR, "valid.jsonl"),
        "test": os.path.join(DATA_DIR, "test.jsonl"),
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    def preprocess_function(examples):
        inputs = [PREFIX + ex for ex in examples["zh"]]
        targets = [ex for ex in examples["en"]]

        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
        labels = tokenizer(
            text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True
        )

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=8,
        load_from_cache_file=False,
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        if np.random.rand() < 0.05:
            print(f"\n[Sample] Pred: {decoded_preds[0]} | Ref: {decoded_labels[0]}")

        result = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels], tokenize="13a")
        return {"bleu": result.score}

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        fp16=False,
        bf16=True,
        dataloader_num_workers=8,
        logging_dir="./logs",
        disable_tqdm=True,
        logging_steps=100,
        logging_strategy="steps",
        report_to="none",
        group_by_length=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting Optimized Training...")
    trainer.train()

    print(f"Saving best model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)

    print("\nRunning evaluation on Test Set...")
    test_results = trainer.predict(tokenized_datasets["test"])
    print(f"Final Test BLEU: {test_results.metrics['test_bleu']:.2f}")

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(test_results.metrics, f, indent=4)


if __name__ == "__main__":
    run_training()
