import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_PATH = "../nlp_llm_project_dev2/mt5_translation_ft"
DATA_PATH = "./data/test.jsonl"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEN_KWARGS = {
    "max_length": 128,
    "num_beams": 4,
    "early_stopping": True,
    "no_repeat_ngram_size": 2,
}

PREFIX = "translate Chinese to English: "


def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def interactive_mode(tokenizer, model):
    print("\n" + "=" * 50)
    print("Interactive Translation Mode")
    print("Type 'q' or 'exit' to quit.")
    print("=" * 50)

    while True:
        try:
            source_text = input("\n[Chinese]: ").strip()

            if source_text.lower() in ["q", "exit", "quit"]:
                print("Bye!")
                break

            if not source_text:
                continue

            input_text = PREFIX + source_text
            inputs = tokenizer(
                input_text, return_tensors="pt", max_length=128, truncation=True
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(**inputs, **GEN_KWARGS)

            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"[English]: {translation}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during translation: {e}")


if __name__ == "__main__":
    # Load model
    tokenizer, model = load_model()

    # interactive mode
    interactive_mode(tokenizer, model)
