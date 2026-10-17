"""Train and run a small SMS spam classifier from the UCI SMS Spam Collection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fraud_simulator import (
    FRAUD_MESSAGES,
    LEGITIMATE_MESSAGES,
    generate_fraud_messages,
    generate_legitimate_messages,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "SMSSpamCollection"
DEFAULT_MODEL = ROOT / "models" / "sms_spam.joblib"
DEFAULT_FEEDBACK = ROOT / "data" / "screening_feedback.jsonl"
DEFAULT_SPAM_THRESHOLD = 0.45
AUTO_TRAIN_EXAMPLES_PER_CLASS = 60
LABELS = {"ham", "spam"}


def load_messages(path: Path) -> tuple[list[str], list[str]]:
    """Read UCI's tab-separated label/message records."""
    texts: list[str] = []
    labels: list[str] = []

    with path.open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                label, text = line.split("\t", maxsplit=1)
            except ValueError as error:
                raise ValueError(f"Malformed dataset row at line {line_number}") from error
            if label not in LABELS or not text.strip():
                raise ValueError(f"Invalid label or empty message at line {line_number}")
            labels.append(label)
            texts.append(text)

    if not texts:
        raise ValueError(f"No messages found in {path}")
    return texts, labels


def load_reviewed_feedback(path: Path) -> tuple[list[str], list[str]]:
    """Load only explicitly approved feedback as spam/ham training examples."""
    texts: list[str] = []
    labels: list[str] = []
    feedback_labels = {"fraud": "spam", "legitimate": "ham"}

    if not path.exists():
        return texts, labels

    with path.open(encoding="utf-8") as feedback_file:
        for line_number, line in enumerate(feedback_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed feedback record at line {line_number}") from error
            if not isinstance(record, dict) or record.get("approved") is not True:
                continue

            message = record.get("message")
            label = feedback_labels.get(record.get("user_label"))
            if not isinstance(message, str) or not message.strip() or label is None:
                raise ValueError(f"Invalid approved feedback record at line {line_number}")
            texts.append(message)
            labels.append(label)

    return texts, labels


def load_user_feedback(path: Path) -> tuple[list[str], list[str]]:
    """Load every valid user-labeled example, including older pending records."""
    texts: list[str] = []
    labels: list[str] = []
    feedback_labels = {"fraud": "spam", "legitimate": "ham"}

    if not path.exists():
        return texts, labels

    with path.open(encoding="utf-8") as feedback_file:
        for line_number, line in enumerate(feedback_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed feedback record at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Invalid feedback record at line {line_number}")

            message = record.get("message")
            if message is None:
                continue
            label = feedback_labels.get(record.get("user_label"))
            if not isinstance(message, str) or not message.strip() or label is None:
                raise ValueError(f"Invalid feedback record at line {line_number}")
            texts.append(message)
            labels.append(label)

    return texts, labels


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def predict_messages(
    model: Any,
    messages: list[str],
    spam_threshold: float = DEFAULT_SPAM_THRESHOLD,
) -> tuple[list[str], list[float]]:
    """Classify messages using an explicit spam-score decision threshold."""
    if not 0 <= spam_threshold <= 1:
        raise ValueError("Spam threshold must be between 0 and 1")
    if not messages:
        return [], []

    spam_index = list(model.classes_).index("spam")
    spam_scores = [float(row[spam_index]) for row in model.predict_proba(messages)]
    predictions = ["spam" if score >= spam_threshold else "ham" for score in spam_scores]
    return predictions, spam_scores


def train(
    data_path: Path,
    model_path: Path,
    feedback_path: Path = DEFAULT_FEEDBACK,
    include_reviewed_feedback: bool = False,
) -> None:
    texts, labels = load_messages(data_path)
    if include_reviewed_feedback:
        feedback_texts, feedback_labels = load_reviewed_feedback(feedback_path)
        texts.extend(feedback_texts)
        labels.extend(feedback_labels)
        print(f"Included {len(feedback_texts)} approved feedback examples.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    evaluation_model = create_pipeline()
    evaluation_model.fit(train_texts, train_labels)
    predictions, _ = predict_messages(evaluation_model, test_texts)
    print("Held-out evaluation (stratified 80/20 split):")
    print(classification_report(test_labels, predictions, labels=["ham", "spam"], zero_division=0))

    final_model = create_pipeline()
    final_model.fit(texts, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"Trained on {len(texts)} messages; model saved to {model_path}")


def train_from_user_feedback(
    data_path: Path,
    feedback_path: Path,
    model_path: Path,
) -> tuple[Pipeline, int]:
    """Retrain and atomically save a model incorporating every user label."""
    texts, labels = load_messages(data_path)
    simulator_count = min(
        AUTO_TRAIN_EXAMPLES_PER_CLASS,
        len(FRAUD_MESSAGES),
        len(LEGITIMATE_MESSAGES),
    )
    texts.extend(generate_fraud_messages(len(FRAUD_MESSAGES), seed=42)[:simulator_count])
    labels.extend(["spam"] * simulator_count)
    texts.extend(generate_legitimate_messages(len(LEGITIMATE_MESSAGES), seed=43)[:simulator_count])
    labels.extend(["ham"] * simulator_count)

    feedback_texts, feedback_labels = load_user_feedback(feedback_path)
    texts.extend(feedback_texts)
    labels.extend(feedback_labels)

    updated_model = create_pipeline()
    updated_model.fit(texts, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_path = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(updated_model, temporary_model_path)
        os.replace(temporary_model_path, model_path)
    finally:
        temporary_model_path.unlink(missing_ok=True)

    return updated_model, len(feedback_texts)


def predict(model_path: Path, message: str) -> None:
    if not message.strip():
        raise ValueError("Message must not be empty")
    model = joblib.load(model_path)
    predictions, scores = predict_messages(model, [message])
    prediction = predictions[0]
    spam_probability = scores[0]
    print(f"Prediction: {prediction}")
    print(f"Spam probability: {spam_probability:.3f}")
    print(f"Decision threshold: {DEFAULT_SPAM_THRESHOLD:.0%}")
    print("Treat this as a screening signal, not proof that a message is fraudulent.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="evaluate and train the model")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    train_parser.add_argument(
        "--include-reviewed-feedback",
        action="store_true",
        help="include only feedback records manually marked approved",
    )

    predict_parser = commands.add_parser("predict", help="classify one SMS message")
    predict_parser.add_argument("message")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)

    args = parser.parse_args()
    try:
        if args.command == "train":
            train(
                args.data,
                args.model,
                feedback_path=args.feedback,
                include_reviewed_feedback=args.include_reviewed_feedback,
            )
        else:
            predict(args.model, args.message)
    except (OSError, ValueError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
