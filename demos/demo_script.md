# Self-Healing Classification System - Demo Script

## Video Demo Timeline (2-4 minutes)

### 0:00 - 0:30: Introduction
- **What**: Self-Healing Classification System using LangGraph DAG orchestration
- **Problem**: ML models make low-confidence predictions that need human verification
- **Solution**: Automated confidence checking with dual fallback strategies

### 0:30 - 1:00: Architecture Overview
Show diagram/explain:
1. **InferenceNode**: DistilBERT with LoRA fine-tuning → predictions + probabilities
2. **ConfidenceCheckNode**: Threshold-based decision (≥75% accept, 50-75% clarify, <50% escalate)
3. **FallbackNode**: 
   - Strategy 1: Human-in-the-loop clarification via CLI
   - Strategy 2: Zero-shot backup model (BART-MNLI)
4. **FinalDecisionNode**: Logging and response

### 1:00 - 2:30: Live CLI Demo

#### Example 1: High Confidence (Direct Accept)
```bash
> This movie was absolutely amazing! Best film I've seen all year!
[InferenceNode] Predicted label: positive | Confidence: 96%
[ConfidenceCheckNode] Confidence HIGH - Direct accept
Final Label: positive (via direct_prediction)
```

#### Example 2: Medium Confidence (User Clarification)
```bash
> The movie was okay, nothing special.
[InferenceNode] Predicted label: positive | Confidence: 62%
[ConfidenceCheckNode] Confidence MEDIUM - Triggering fallback...
[FallbackNode] Could you clarify? Was this a negative review? (yes/no)
> yes
Final Label: negative (Corrected via user_clarification)
```

#### Example 3: Low Confidence (Backup Model Escalation)
```bash
> I'm not sure what to think about this film.
[InferenceNode] Predicted label: positive | Confidence: 48%
[ConfidenceCheckNode] Confidence LOW - Escalating to backup model...
[FallbackNode] Running zero-shot classification...
Backup Model: negative (72% confidence)
Final Label: negative (via backup_model_escalation)
```

### 2:30 - 3:00: Show Logs & Metrics
```bash
tail -f logs/app.jsonl
```

Show JSON log entry with:
- Timestamp, request_id
- Input text, predictions, confidence
- Fallback activation and strategy
- User response (if applicable)
- Final decision

### 3:00 - 3:30: Wrap Up

**Key Features Demonstrated:**
1. ✅ LoRA fine-tuned DistilBERT for efficient training
2. ✅ LangGraph DAG orchestration with conditional routing
3. ✅ Confidence-based fallback mechanism
4. ✅ Human-in-the-loop clarification
5. ✅ Zero-shot backup model for escalation
6. ✅ Structured JSON logging for all decisions
7. ✅ Interactive CLI with rich formatting

**How to Reproduce:**
```bash
make install          # Install dependencies
make train           # Train model (5K samples)
make run-cli         # Start interactive CLI
make test            # Run unit tests
```

## Technical Highlights for Reviewers

1. **Parameter-Efficient Training**: LoRA reduces trainable params by ~99%
2. **Temperature Scaling**: Improves probability calibration (ECE metric)
3. **Dual Fallback**: Combines human intelligence + backup AI model
4. **Production-Ready**: Docker, Makefile, comprehensive tests
5. **Observability**: Structured logs in JSON format for analytics
