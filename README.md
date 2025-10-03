# Self-Healing Classification System

Industry-level machine learning classification system with LangGraph DAG orchestration, confidence-based fallback mechanisms, and human-in-the-loop clarification.

## 🎯 Features

- **LoRA Fine-Tuned DistilBERT**: Parameter-efficient training with PEFT (1% trainable parameters)
- **LangGraph DAG Orchestration**: Conditional workflow with 4 nodes (Inference → Confidence Check → Fallback → Final Decision)
- **Confidence-Based Fallback**: 
  - ≥75% confidence → Direct acceptance
  - 50-75% confidence → User clarification
  - <50% confidence → Escalation to backup model
- **Dual Fallback Strategies**:
  1. Human-in-the-loop CLI clarification
  2. Zero-shot backup model (BART-MNLI)
- **Structured JSON Logging**: Complete audit trail with timestamps, predictions, confidence scores, and fallback events
- **Temperature Scaling**: Improved probability calibration for better confidence estimates
- **Production-Ready**: Docker, Makefile, comprehensive unit tests, CI/CD ready

## 🚀 Quick Start

### 1. Install Dependencies

```bash
make install
# or
pip install -r requirements.txt
```

### 2. Setup Model

Use pre-trained model (fastest):
```bash
python setup_pretrained_model.py
```

Or train from scratch with LoRA:
```bash
make train           # Quick training (2K samples)
make train-full      # Full dataset training
```

### 3. Run the CLI

Interactive mode (with human clarification):
```bash
make run-cli
# or
python -m src.app.cli run
```

Non-interactive mode (backup model only):
```bash
python -m src.app.cli run --non-interactive
```

### 4. Run Demo

```bash
python run_demo.py
```

## 📊 Architecture

```
User Input
    ↓
┌─────────────────┐
│ InferenceNode   │ → DistilBERT inference → probabilities
└────────┬────────┘
         ↓
┌─────────────────┐
│ ConfidenceNode  │ → Check threshold → accept/clarify/escalate
└────────┬────────┘
         ↓
    ┌────┴────┐
    │ action? │
    └────┬────┘
         ↓
┌─────────────────┐
│  FallbackNode   │ → Strategy 1: Ask user OR Strategy 2: Backup model
└────────┬────────┘
         ↓
┌─────────────────┐
│ FinalDecision   │ → Log decision → Return result
└─────────────────┘
```

## 🧪 Testing

Run all tests:
```bash
make test
# or
pytest tests/ -v
```

Run specific tests:
```bash
pytest tests/test_nodes.py -v          # Test individual nodes
pytest tests/test_dag_flow.py -v       # Test DAG integration
```

## 📝 Logging & Monitoring

View recent logs:
```bash
make logs              # Last 20 text logs
make logs-json         # Last 10 JSON logs
```

JSON log structure:
```json
{
  "timestamp": "2025-10-03T12:00:00",
  "request_id": "uuid",
  "input_text": "...",
  "inference": {
    "pred_label": "positive",
    "probs": {"positive": 0.85, "negative": 0.15},
    "confidence": 0.85
  },
  "confidence_check": {
    "threshold_accept": 0.75,
    "status": "HIGH"
  },
  "fallback": {
    "activated": false
  },
  "final_decision": {
    "label": "positive",
    "via": "direct_prediction"
  }
}
```

## 🐳 Docker

Build and run:
```bash
make docker-build
docker run -it self-healing-classifier
```

## 📁 Project Structure

```
.
├── src/app/
│   ├── nodes/              # LangGraph DAG nodes
│   │   ├── inference_node.py
│   │   ├── confidence_node.py
│   │   ├── fallback_node.py
│   │   └── final_decision_node.py
│   ├── model/              # Training & calibration
│   │   ├── trainer.py
│   │   └── temperature_scaling.py
│   ├── dag.py              # LangGraph orchestrator
│   ├── cli.py              # Typer CLI interface
│   ├── config.py           # Configuration
│   └── logger.py           # Structured logging
├── tests/                  # Unit & integration tests
├── checkpoints/            # Model checkpoints
├── logs/                   # Application logs
└── demos/                  # Demo scripts
```

## ⚙️ Configuration

Edit `src/app/config.py` to customize:

```python
CONFIDENCE_THRESHOLDS = {
    "accept": 0.75,      # High confidence threshold
    "clarify": 0.50,     # Medium confidence threshold
}

LORA_CONFIG = {
    "r": 8,              # LoRA rank
    "lora_alpha": 32,    # LoRA scaling
    "lora_dropout": 0.1,
}
```

## 🎬 Demo Script

See `demos/demo_script.md` for a complete demo walkthrough including:
- Architecture explanation
- Live CLI demonstration
- Example fallback scenarios
- Log inspection

## 🔧 Makefile Commands

```bash
make install           # Install dependencies
make train             # Train model (2K samples)
make train-full        # Train on full dataset
make run-cli           # Run interactive CLI
make logs              # View logs
make test              # Run tests
make clean             # Clean checkpoints and logs
make docker-build      # Build Docker image
```

## 📈 Performance Metrics

With LoRA fine-tuning:
- **Trainable Parameters**: 739K (1.09% of total)
- **Training Speed**: ~3 epochs in <10 minutes (2K samples, CPU)
- **Inference**: <200ms per prediction
- **Calibration**: ECE improvement with temperature scaling

## 🛠️ Technology Stack

**Core ML:**
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Accelerate

**Orchestration:**
- LangGraph
- LangChain Core

**Interface & Logging:**
- Typer (CLI)
- Structlog (JSON logging)
- Rich (terminal formatting)

**Testing:**
- Pytest
- Pytest-asyncio

## 🎯 Use Cases

1. **Content Moderation**: High-stakes decisions with human verification
2. **Customer Support**: Sentiment analysis with unclear cases escalated
3. **Medical Diagnosis**: Critical predictions requiring confirmation
4. **Financial Risk**: Automated decisions with human oversight on edge cases

## 📚 Next Steps

1. **Production Deployment**: Use the provided Dockerfile
2. **Model Upload**: Push trained model to Hugging Face Hub
3. **Dashboard**: Add Streamlit UI for analytics
4. **CI/CD**: Configure GitHub Actions for testing
5. **Auto-Relabeling**: Store clarified examples for model improvement

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run `make test` to verify
5. Submit a pull request

---

Built with ❤️ using LangGraph, LoRA, and Human-in-the-Loop AI
