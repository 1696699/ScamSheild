"""Web interface for screening SMS messages with the trained spam model."""

import argparse
import secrets
from pathlib import Path
from typing import Any

import joblib
from flask import Flask, jsonify, render_template, request

from fraud_simulator import FRAUD_MESSAGES, LEGITIMATE_MESSAGES
from sms_spam import (
    DEFAULT_MODEL,
    DEFAULT_SPAM_THRESHOLD,
    predict_messages,
)

MAX_MESSAGE_LENGTH = 5000


def create_app(
    model_path: Path = DEFAULT_MODEL,
    model: Any = None,
    spam_threshold: float = DEFAULT_SPAM_THRESHOLD,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    classifier = model if model is not None else joblib.load(model_path)
    sample_pools = {
        "fraud": list(FRAUD_MESSAGES),
        "legitimate": list(LEGITIMATE_MESSAGES),
    }
    generated_samples: dict[str, set[str]] = {
        sample_type: set() for sample_type in sample_pools
    }

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/api/generate-sample")
    def generate_sample() -> Any:
        payload = request.get_json(silent=True)
        sample_type = payload.get("sample_type", "fraud") if isinstance(payload, dict) else "fraud"
        if not isinstance(sample_type, str) or sample_type not in sample_pools:
            return jsonify(error="Choose a fraudulent or legitimate sample."), 400

        sample_pool = sample_pools[sample_type]
        generated_for_type = generated_samples[sample_type]
        available_samples = [sample for sample in sample_pool if sample not in generated_for_type]
        if not available_samples:
            generated_for_type.clear()
            available_samples = sample_pool
        message = secrets.choice(available_samples)
        generated_for_type.add(message)
        prediction, spam_scores = predict_messages(classifier, [message], spam_threshold)
        predicted_status = "fraudulent" if prediction[0] == "spam" else "non-fraudulent"
        expected_status = "fraudulent" if sample_type == "fraud" else "non-fraudulent"
        return jsonify(
            message=message,
            notice="SIMULATED - DO NOT SEND. Fictional sample; links use the reserved .invalid domain.",
            sample_type=sample_type,
            ground_truth=expected_status,
            prediction=prediction[0],
            detector_verdict=predicted_status,
            spam_score=spam_scores[0],
            spam_threshold=spam_threshold,
            matches_ground_truth=predicted_status == expected_status,
        ), 200

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

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--spam-threshold", type=float, default=DEFAULT_SPAM_THRESHOLD)
    args = parser.parse_args()
    create_app(
        model_path=args.model,
        spam_threshold=args.spam_threshold,
    ).run(
        host="127.0.0.1", port=args.port, debug=False
    )