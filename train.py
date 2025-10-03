#!/usr/bin/env python
import sys
import argparse
from src.app.model.trainer import train_model
from src.app.config import Config

def main():
    parser = argparse.ArgumentParser(description='Train the self-healing classifier')
    parser.add_argument('--max-samples', type=int, default=5000, 
                       help='Maximum number of training samples (default: 5000, use None for full dataset)')
    parser.add_argument('--full', action='store_true',
                       help='Train on full dataset')
    
    args = parser.parse_args()
    
    max_samples = None if args.full else args.max_samples
    
    print(f"Starting training with max_samples={max_samples}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Dataset: {Config.DATASET_NAME}")
    print(f"LoRA config: {Config.LORA_CONFIG}")
    print()
    
    trainer_obj, eval_results = train_model(max_samples=max_samples)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Model saved to: {Config.CHECKPOINTS_DIR / 'model'}")
    print(f"Evaluation results: {eval_results}")
    print("="*50)

if __name__ == "__main__":
    main()
