#!/usr/bin/env python
import sys
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.app.model.trainer import train_model
from src.app.config import Config

print("="*60)
print("SELF-HEALING CLASSIFIER - TRAINING")
print("="*60)
print(f"Model: {Config.MODEL_NAME}")
print(f"Dataset: {Config.DATASET_NAME}")
print(f"Training samples: 2000 (quick demo)")
print(f"LoRA r={Config.LORA_CONFIG['r']}, alpha={Config.LORA_CONFIG['lora_alpha']}")
print("="*60)
print()

try:
    trainer_obj, eval_results = train_model(max_samples=2000)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model saved to: {Config.CHECKPOINTS_DIR / 'model'}")
    print(f"\nEvaluation Results:")
    for metric, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run the CLI: python -m src.app.cli run")
    print("  2. Or use make: make run-cli")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error during training: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
