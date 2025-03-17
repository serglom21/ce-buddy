import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def generate_sentry_suggestion(input_text, model, tokenizer):
    """Generates a Sentry suggestion for the given input text."""
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_text>")
        sys.exit(1)

    input_text = sys.argv[1]

    try:
        # Load the fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("./results")
        model = AutoModelForSeq2SeqLM.from_pretrained("./results")

        try:
            suggestion = generate_sentry_suggestion(input_text, model, tokenizer)
            print("Sentry Suggestion:", suggestion)
        except Exception as e:
            print(f"Error: {e}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)