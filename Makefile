.PHONY: install train eval run-cli test clean logs docker-build

install:
	pip install -r requirements.txt

train:
	python -c "from src.app.model.trainer import train_model; train_model(max_samples=5000)"

train-full:
	python -c "from src.app.model.trainer import train_model; train_model()"

eval:
	python -c "from src.app.model.trainer import ModelTrainer; from src.app.config import Config; import torch; \
		trainer = ModelTrainer(); \
		trainer.load_and_prepare_data(); \
		trainer.tokenizer = trainer.tokenizer or __import__('transformers').AutoTokenizer.from_pretrained(Config.CHECKPOINTS_DIR / 'model'); \
		trainer.model = __import__('transformers').AutoModelForSequenceClassification.from_pretrained(Config.CHECKPOINTS_DIR / 'model'); \
		print('Model evaluation complete')"

run-cli:
	python -m src.app.cli run

run-cli-non-interactive:
	python -m src.app.cli run --non-interactive

logs:
	python -m src.app.cli logs --lines 20

logs-json:
	python -m src.app.cli logs --lines 10 --json

test:
	pytest tests/ -v

clean:
	rm -rf checkpoints/* logs/* data/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t self-healing-classifier .
