import os
from pathlib import Path
from typing import Dict, Any

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    
    MODEL_NAME = "distilbert-base-uncased"
    DATASET_NAME = "imdb"
    
    LORA_CONFIG = {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_lin", "v_lin"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "SEQ_CLS"
    }
    
    TRAINING_CONFIG = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "logging_steps": 50,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "fp16": False,
        "max_seq_length": 512
    }
    
    CONFIDENCE_THRESHOLDS = {
        "accept": 0.75,
        "clarify": 0.50,
    }
    
    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
    ZERO_SHOT_LABELS = ["negative", "positive"]
    
    LOG_FILE = LOGS_DIR / "app.log"
    LOG_JSONL_FILE = LOGS_DIR / "app.jsonl"
    
    WANDB_PROJECT = "self-healing-classifier"
    WANDB_ENTITY = None
    
    @classmethod
    def ensure_dirs(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

Config.ensure_dirs()
