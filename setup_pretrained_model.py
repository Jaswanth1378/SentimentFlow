#!/usr/bin/env python
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
output_dir = Path("checkpoints/model")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Setting up pre-trained sentiment model for demo")
print("="*60)
print(f"Model: {model_name}")
print(f"Output: {output_dir}")
print()

print("Downloading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("Saving to checkpoints/model...")
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

print("\n" + "="*60)
print("✅ Model setup complete!")
print("="*60)
print(f"Model saved to: {output_dir}")
print("\nNext steps:")
print("  Run the CLI: python -m src.app.cli run")
print("="*60)
