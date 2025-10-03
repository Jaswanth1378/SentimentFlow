import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import structlog
from src.app.config import Config

class StructuredLogger:
    def __init__(self):
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )
        self.logger = structlog.get_logger()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.file_logger = logging.getLogger(__name__)
    
    def log_inference(
        self,
        request_id: str,
        input_text: str,
        pred_label: str,
        probs: Dict[str, float],
        confidence: float,
        confidence_status: str,
        fallback_activated: bool = False,
        fallback_strategy: Optional[str] = None,
        fallback_question: Optional[str] = None,
        user_response: Optional[str] = None,
        final_label: Optional[str] = None,
        final_decision_via: Optional[str] = None
    ):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "input_text": input_text,
            "inference": {
                "pred_label": pred_label,
                "probs": probs,
                "confidence": confidence
            },
            "confidence_check": {
                "threshold_accept": Config.CONFIDENCE_THRESHOLDS["accept"],
                "threshold_clarify": Config.CONFIDENCE_THRESHOLDS["clarify"],
                "status": confidence_status
            },
            "fallback": {
                "activated": fallback_activated,
                "strategy": fallback_strategy,
                "question": fallback_question,
                "user_response": user_response
            } if fallback_activated else {"activated": False},
            "final_decision": {
                "label": final_label or pred_label,
                "via": final_decision_via or "direct_prediction"
            }
        }
        
        with open(Config.LOG_JSONL_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.file_logger.info(f"Request {request_id}: {final_label or pred_label} (confidence: {confidence:.2%})")
        
        return log_entry

logger_instance = StructuredLogger()
