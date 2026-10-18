"""Train and run a small SMS spam classifier from the UCI SMS Spam Collection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
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
DEFAULT_SMISHING_DATA = ROOT / "data" / "Smishing-Dataset-IMC25" / "final_dataset_output.csv"
DEFAULT_MENDELEY_DATA = ROOT / "data" / "Mendeley-SMS-Phishing" / "Dataset_5971.csv"
DEFAULT_NUS_REVIEWED_DATA = ROOT / "data" / "NUS-SMS-Corpus" / "review_queue.csv"
DEFAULT_MULTILINGUAL_DATA = (
    ROOT / "data" / "SMS-Scam-Detection-Multilingual"
    / "sms_scam_detection_dataset_merged_with_lang.csv"
)
DEFAULT_MODEL = ROOT / "models" / "sms_spam.joblib"
DEFAULT_FEEDBACK = ROOT / "data" / "screening_feedback.jsonl"
DEFAULT_SPAM_THRESHOLD = 0.45
AUTO_TRAIN_EXAMPLES_PER_CLASS = 200
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


def load_smishing_messages(path: Path) -> tuple[list[str], list[str]]:
    """Load smishing reports as spam, using English translations when provided."""
    import csv

    texts: list[str] = []
    labels: list[str] = []

    with path.open(encoding="utf-8-sig", newline="") as dataset:
        reader = csv.DictReader(dataset)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"Expected a 'text' column in smishing dataset {path}")

        for line_number, row in enumerate(reader, start=2):
            original = (row.get("text") or "").strip()
            translation = (row.get("translation") or "").strip()
            message = translation or original
            if not message:
                raise ValueError(f"Empty message at line {line_number} in {path}")
            texts.append(message)
            labels.append("spam")

    if not texts:
        raise ValueError(f"No messages found in smishing dataset {path}")
    return texts, labels


def load_mendeley_messages(path: Path) -> tuple[list[str], list[str]]:
    """Load Mendeley ham/spam/smishing records as binary ham/spam examples."""
    import csv

    texts: list[str] = []
    labels: list[str] = []
    label_map = {"ham": "ham", "spam": "spam", "smishing": "spam"}

    with path.open(encoding="utf-8-sig", newline="") as dataset:
        reader = csv.DictReader(dataset)
        if not reader.fieldnames or not {"LABEL", "TEXT"}.issubset(reader.fieldnames):
            raise ValueError(f"Expected 'LABEL' and 'TEXT' columns in dataset {path}")

        for line_number, row in enumerate(reader, start=2):
            message = (row.get("TEXT") or "").strip()
            label = label_map.get((row.get("LABEL") or "").strip().casefold())
            if label is None or not message:
                raise ValueError(f"Invalid label or empty message at line {line_number} in {path}")
            texts.append(message)
            labels.append(label)

    if not texts:
        raise ValueError(f"No messages found in Mendeley dataset {path}")
    return texts, labels


def load_reviewed_nus_messages(path: Path) -> tuple[list[str], list[str]]:
    """Load only NUS messages with an explicit human-reviewed binary label."""
    import csv

    texts: list[str] = []
    labels: list[str] = []

    if not path.exists():
        return texts, labels

    with path.open(encoding="utf-8-sig", newline="") as dataset:
        reader = csv.DictReader(dataset)
        required_columns = {"text", "reviewed_label"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Expected columns {', '.join(sorted(required_columns))} in NUS review file {path}"
            )

        for line_number, row in enumerate(reader, start=2):
            label = (row.get("reviewed_label") or "").strip().casefold()
            if not label:
                continue
            message = (row.get("text") or "").strip()
            if label not in LABELS or not message:
                raise ValueError(f"Invalid reviewed label or empty message at line {line_number} in {path}")
            texts.append(message)
            labels.append(label)

    return texts, labels


def load_multilingual_messages(path: Path) -> tuple[list[str], list[str]]:
    """Load multilingual labeled ham/spam examples, ignoring auxiliary columns."""
    import csv

    texts: list[str] = []
    labels: list[str] = []

    with path.open(encoding="utf-8-sig", newline="") as dataset:
        reader = csv.DictReader(dataset)
        if not reader.fieldnames or not {"label", "text"}.issubset(reader.fieldnames):
            raise ValueError(f"Expected 'label' and 'text' columns in dataset {path}")

        for line_number, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip().casefold()
            message = (row.get("text") or "").strip()
            if label not in LABELS:
                raise ValueError(f"Invalid label at line {line_number} in {path}")
            if not message:
                warnings.warn(
                    f"Skipping empty message at line {line_number} in multilingual dataset {path}",
                    stacklevel=2,
                )
                continue
            texts.append(message)
            labels.append(label)

    if not texts:
        raise ValueError(f"No messages found in multilingual dataset {path}")
    return texts, labels


def load_training_messages(
    data_path: Path,
    smishing_data_path: Path | None = DEFAULT_SMISHING_DATA,
    mendeley_data_path: Path | None = DEFAULT_MENDELEY_DATA,
    nus_reviewed_data_path: Path | None = DEFAULT_NUS_REVIEWED_DATA,
    multilingual_data_path: Path | None = DEFAULT_MULTILINGUAL_DATA,
) -> tuple[list[str], list[str]]:
    """Combine message corpora and drop duplicates with conflicting labels."""
    texts, labels = load_messages(data_path)
    if smishing_data_path is not None:
        smishing_texts, smishing_labels = load_smishing_messages(smishing_data_path)
        texts.extend(smishing_texts)
        labels.extend(smishing_labels)
    if mendeley_data_path is not None:
        mendeley_texts, mendeley_labels = load_mendeley_messages(mendeley_data_path)
        texts.extend(mendeley_texts)
        labels.extend(mendeley_labels)
    if nus_reviewed_data_path is not None:
        nus_texts, nus_labels = load_reviewed_nus_messages(nus_reviewed_data_path)
        texts.extend(nus_texts)
        labels.extend(nus_labels)
    if multilingual_data_path is not None:
        multilingual_texts, multilingual_labels = load_multilingual_messages(multilingual_data_path)
        texts.extend(multilingual_texts)
        labels.extend(multilingual_labels)

    unique_messages: dict[str, tuple[str, str]] = {}
    conflicting_messages: set[str] = set()
    for text, label in zip(texts, labels):
        key = " ".join(text.casefold().split())
        existing = unique_messages.get(key)
        if existing is not None and existing[1] != label:
            conflicting_messages.add(key)
        else:
            unique_messages.setdefault(key, (text, label))

    for key in conflicting_messages:
        unique_messages.pop(key, None)
    if not unique_messages:
        raise ValueError("No unambiguous labeled messages available for training")

    return (
        [message for message, _ in unique_messages.values()],
        [label for _, label in unique_messages.values()],
    )


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
    smishing_data_path: Path | None = DEFAULT_SMISHING_DATA,
    mendeley_data_path: Path | None = DEFAULT_MENDELEY_DATA,
    nus_reviewed_data_path: Path | None = DEFAULT_NUS_REVIEWED_DATA,
    multilingual_data_path: Path | None = DEFAULT_MULTILINGUAL_DATA,
) -> None:
    texts, labels = load_training_messages(
        data_path,
        smishing_data_path,
        mendeley_data_path,
        nus_reviewed_data_path,
        multilingual_data_path,
    )
    print(f"Training corpus contains {len(texts)} unique messages.")
    if include_reviewed_feedback:
        feedback_texts, feedback_labels = load_reviewed_feedback(feedback_path)
        texts.extend(feedback_texts)
        labels.extend(feedback_labels)
        print(f"Included {len(feedback_texts)} approved feedback examples.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    simulator_scams = generate_fraud_messages(AUTO_TRAIN_EXAMPLES_PER_CLASS, seed=42)
    simulator_legitimate = generate_legitimate_messages(AUTO_TRAIN_EXAMPLES_PER_CLASS, seed=43)
    evaluation_model = create_pipeline()
    evaluation_model.fit(
        train_texts + simulator_scams + simulator_legitimate,
        train_labels
        + ["spam"] * len(simulator_scams)
        + ["ham"] * len(simulator_legitimate),
    )
    predictions, _ = predict_messages(evaluation_model, test_texts)
    print(
        "Held-out evaluation (stratified 80/20 split; "
        f"{len(simulator_scams) + len(simulator_legitimate)} synthetic training examples):"
    )
    print(classification_report(test_labels, predictions, labels=["ham", "spam"], zero_division=0))

    final_model = create_pipeline()
    final_model.fit(
        texts + simulator_scams + simulator_legitimate,
        labels
        + ["spam"] * len(simulator_scams)
        + ["ham"] * len(simulator_legitimate),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    print(
        f"Trained on {len(texts)} unique corpus/feedback messages and "
        f"{len(simulator_scams) + len(simulator_legitimate)} synthetic examples; "
        f"model saved to {model_path}"
    )


def train_from_user_feedback(
    data_path: Path,
    feedback_path: Path,
    model_path: Path,
    smishing_data_path: Path | None = DEFAULT_SMISHING_DATA,
    mendeley_data_path: Path | None = DEFAULT_MENDELEY_DATA,
    nus_reviewed_data_path: Path | None = DEFAULT_NUS_REVIEWED_DATA,
    multilingual_data_path: Path | None = DEFAULT_MULTILINGUAL_DATA,
) -> tuple[Pipeline, int]:
    """Retrain and atomically save a model incorporating every user label."""
    texts, labels = load_training_messages(
        data_path,
        smishing_data_path,
        mendeley_data_path,
        nus_reviewed_data_path,
        multilingual_data_path,
    )
    simulator_count = min(
        AUTO_TRAIN_EXAMPLES_PER_CLASS,
        len(FRAUD_MESSAGES),
        len(LEGITIMATE_MESSAGES),
    )
    texts.extend(generate_fraud_messages(simulator_count, seed=42))
    labels.extend(["spam"] * simulator_count)
    texts.extend(generate_legitimate_messages(simulator_count, seed=43))
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
    train_parser.add_argument("--smishing-data", type=Path, default=DEFAULT_SMISHING_DATA)
    train_parser.add_argument("--mendeley-data", type=Path, default=DEFAULT_MENDELEY_DATA)
    train_parser.add_argument("--nus-reviewed-data", type=Path, default=DEFAULT_NUS_REVIEWED_DATA)
    train_parser.add_argument("--multilingual-data", type=Path, default=DEFAULT_MULTILINGUAL_DATA)
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
                smishing_data_path=args.smishing_data,
                mendeley_data_path=args.mendeley_data,
                nus_reviewed_data_path=args.nus_reviewed_data,
                multilingual_data_path=args.multilingual_data,
            )
        else:
            predict(args.model, args.message)
    except (OSError, ValueError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
