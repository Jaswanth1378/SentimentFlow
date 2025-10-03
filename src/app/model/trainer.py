import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
from src.app.config import Config
import os

class ModelTrainer:
    def __init__(self, model_name: str = None, dataset_name: str = None):
        self.model_name = model_name or Config.MODEL_NAME
        self.dataset_name = dataset_name or Config.DATASET_NAME
        self.tokenizer = None
        self.model = None
        self.dataset = None
    
    def load_and_prepare_data(self, max_samples: int = None):
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)
        
        if max_samples:
            self.dataset["train"] = self.dataset["train"].select(range(min(max_samples, len(self.dataset["train"]))))
            test_samples = min(max_samples // 10, len(self.dataset["test"]))
            self.dataset["test"] = self.dataset["test"].select(range(test_samples))
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                max_length=Config.TRAINING_CONFIG["max_seq_length"]
            )
        
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        
        self.dataset = self.dataset.rename_column("label", "labels")
        
        return self.dataset
    
    def create_model_with_lora(self):
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        lora_config = LoraConfig(
            r=Config.LORA_CONFIG["r"],
            lora_alpha=Config.LORA_CONFIG["lora_alpha"],
            target_modules=Config.LORA_CONFIG["target_modules"],
            lora_dropout=Config.LORA_CONFIG["lora_dropout"],
            bias=Config.LORA_CONFIG["bias"],
            task_type=TaskType.SEQ_CLS
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, output_dir: str = None):
        if output_dir is None:
            output_dir = Config.CHECKPOINTS_DIR / "model"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=Config.TRAINING_CONFIG["num_train_epochs"],
            per_device_train_batch_size=Config.TRAINING_CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=Config.TRAINING_CONFIG["per_device_eval_batch_size"],
            learning_rate=Config.TRAINING_CONFIG["learning_rate"],
            weight_decay=Config.TRAINING_CONFIG["weight_decay"],
            warmup_steps=Config.TRAINING_CONFIG["warmup_steps"],
            logging_dir=str(Config.LOGS_DIR),
            logging_steps=Config.TRAINING_CONFIG["logging_steps"],
            eval_strategy=Config.TRAINING_CONFIG["eval_strategy"],
            save_strategy=Config.TRAINING_CONFIG["save_strategy"],
            load_best_model_at_end=Config.TRAINING_CONFIG["load_best_model_at_end"],
            metric_for_best_model=Config.TRAINING_CONFIG["metric_for_best_model"],
            fp16=Config.TRAINING_CONFIG["fp16"],
            report_to="none",
            save_total_limit=2,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Evaluating on test set...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer, eval_results

def train_model(max_samples: int = None):
    trainer = ModelTrainer()
    trainer.load_and_prepare_data(max_samples=max_samples)
    trainer.create_model_with_lora()
    return trainer.train()
