"""Web interface for screening SMS messages with the trained spam model."""

import argparse
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
from flask import Flask, jsonify, render_template, request

from fraud_simulator import FRAUD_MESSAGES
from sms_spam import (
    DEFAULT_DATA,
    DEFAULT_FEEDBACK,
    DEFAULT_MODEL,
    DEFAULT_SPAM_THRESHOLD,
    predict_messages,
    train_from_user_feedback,
)

MAX_MESSAGE_LENGTH = 5000


def create_app(
    model_path: Path = DEFAULT_MODEL,
    model: Any = None,
    feedback_path: Path = DEFAULT_FEEDBACK,
    spam_threshold: float = DEFAULT_SPAM_THRESHOLD,
    data_path: Path = DEFAULT_DATA,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    classifier = model if model is not None else joblib.load(model_path)
    sample_pool = list(FRAUD_MESSAGES)
    generated_samples: set[str] = set()
    feedback_training_lock = Lock()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/api/generate-sample")
    def generate_sample() -> Any:
        available_samples = [sample for sample in sample_pool if sample not in generated_samples]
        if not available_samples:
            generated_samples.clear()
            available_samples = sample_pool
        message = secrets.choice(available_samples)
        generated_samples.add(message)
        return jsonify(message=message), 200

    @app.post("/api/analyze")
    def analyze() -> tuple[Any, int] | Any:
        payload = request.get_json(silent=True)
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, str) or not message.strip():
            return jsonify(error="Paste a message to screen."), 400
        if len(message) > MAX_MESSAGE_LENGTH:
            return jsonify(error=f"Messages must be {MAX_MESSAGE_LENGTH} characters or fewer."), 413

        predictions, spam_scores = predict_messages(classifier, [message], spam_threshold)
        prediction = predictions[0]
        spam_score = spam_scores[0]
        verdict = "Likely spam" if prediction == "spam" else "No spam detected"
        return jsonify(
            prediction=prediction,
            spam_score=spam_score,
            spam_threshold=spam_threshold,
            verdict=verdict,
        ), 200

    @app.post("/api/feedback")
    def feedback() -> tuple[Any, int] | Any:
        nonlocal classifier
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Feedback must be sent as JSON."), 400

        actual_status = payload.get("actual_status")
        prediction = payload.get("prediction")
        message = payload.get("message")
        if actual_status not in {"fraud", "legitimate"} or prediction not in {"spam", "ham"}:
            return jsonify(error="Choose a valid message status."), 400
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_LENGTH:
            return jsonify(error="A valid message is required for feedback."), 400

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_prediction": prediction,
            "user_label": actual_status,
            "message": message,
            "approved": True,
        }
        with feedback_training_lock:
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with feedback_path.open("a", encoding="utf-8") as feedback_file:
                feedback_file.write(json.dumps(record) + "\n")
            try:
                updated_classifier, feedback_count = train_from_user_feedback(
                    data_path, feedback_path, model_path
                )
            except Exception:
                app.logger.exception("Automatic feedback training failed")
                return jsonify(
                    error="Feedback was saved, but retraining failed. It will be retried with the next rating."
                ), 500
            classifier = updated_classifier

        return jsonify(
            message="Feedback saved and the local classifier was retrained.",
            trained_feedback_examples=feedback_count,
        ), 200

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--spam-threshold", type=float, default=DEFAULT_SPAM_THRESHOLD)
    args = parser.parse_args()
    create_app(model_path=args.model, spam_threshold=args.spam_threshold).run(
        host="127.0.0.1", port=args.port, debug=False
    )