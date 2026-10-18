"""Run a reproducible challenge between synthetic messages and the SMS model."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from fraud_simulator import FRAUD_MESSAGES, LEGITIMATE_MESSAGES, generate_fraud_messages, generate_legitimate_messages
from sms_spam import (
    DEFAULT_DATA,
    DEFAULT_FEEDBACK,
    DEFAULT_MENDELEY_DATA,
    DEFAULT_MULTILINGUAL_DATA,
    DEFAULT_NUS_REVIEWED_DATA,
    DEFAULT_MODEL,
    DEFAULT_SMISHING_DATA,
    DEFAULT_SPAM_THRESHOLD,
    create_pipeline,
    load_training_messages,
    load_reviewed_feedback,
    predict_messages,
)

DEFAULT_CANDIDATE_MODEL = DEFAULT_MODEL.with_name("sms_spam_battle_candidate.joblib")


@dataclass(frozen=True)
class BattleResult:
    examples_per_side: int
    caught_scams: int
    missed_scams: tuple[str, ...]
    false_alarms: tuple[str, ...]

    @property
    def detection_rate(self) -> float:
        return self.caught_scams / self.examples_per_side

    @property
    def false_alarm_rate(self) -> float:
        return len(self.false_alarms) / self.examples_per_side

    @property
    def balanced_score(self) -> float:
        return (self.detection_rate + 1 - self.false_alarm_rate) / 2


def evaluate_messages(
    model: Any,
    scam_messages: list[str],
    legitimate_messages: list[str],
    spam_threshold: float = DEFAULT_SPAM_THRESHOLD,
) -> BattleResult:
    """Score the model against labeled fictional scam and legitimate messages."""
    if not scam_messages or len(scam_messages) != len(legitimate_messages):
        raise ValueError("Battle requires equally sized, non-empty message groups")

    examples_per_side = len(scam_messages)
    predictions, _ = predict_messages(model, scam_messages + legitimate_messages, spam_threshold)

    caught_scams = sum(prediction == "spam" for prediction in predictions[:examples_per_side])
    missed_scams = tuple(
        message
        for message, prediction in zip(scam_messages, predictions[:examples_per_side])
        if prediction != "spam"
    )
    false_alarms = tuple(
        message
        for message, prediction in zip(legitimate_messages, predictions[examples_per_side:])
        if prediction == "spam"
    )
    return BattleResult(examples_per_side, caught_scams, missed_scams, false_alarms)


def evaluate_battle(
    model: Any,
    examples_per_side: int = 25,
    seed: int = 42,
    spam_threshold: float = DEFAULT_SPAM_THRESHOLD,
) -> BattleResult:
    """Score the model against equally sized fictional scam and legitimate batches."""
    scam_messages = generate_fraud_messages(examples_per_side, seed)
    legitimate_messages = generate_legitimate_messages(examples_per_side, seed + 1)
    return evaluate_messages(model, scam_messages, legitimate_messages, spam_threshold)


def split_self_play_examples(
    seed: int = 42,
    holdout_per_class: int = 20,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return disjoint training and holdout examples for each generated class."""
    if holdout_per_class < 1 or holdout_per_class >= min(len(FRAUD_MESSAGES), len(LEGITIMATE_MESSAGES)):
        raise ValueError("Holdout must be between 1 and 99 examples per class")

    scams = generate_fraud_messages(len(FRAUD_MESSAGES), seed)
    legitimate = generate_legitimate_messages(len(LEGITIMATE_MESSAGES), seed + 1)
    return (
        scams[:-holdout_per_class],
        scams[-holdout_per_class:],
        legitimate[:-holdout_per_class],
        legitimate[-holdout_per_class:],
    )


def train_battle_candidate(
    source_model_path: Path = DEFAULT_MODEL,
    data_path: Path = DEFAULT_DATA,
    smishing_data_path: Path | None = DEFAULT_SMISHING_DATA,
    mendeley_data_path: Path | None = DEFAULT_MENDELEY_DATA,
    nus_reviewed_data_path: Path | None = DEFAULT_NUS_REVIEWED_DATA,
    multilingual_data_path: Path | None = DEFAULT_MULTILINGUAL_DATA,
    output_model_path: Path = DEFAULT_CANDIDATE_MODEL,
    feedback_path: Path = DEFAULT_FEEDBACK,
    rounds: int = 3,
    examples_per_round: int = 20,
    holdout_per_class: int = 20,
    seed: int = 42,
    include_reviewed_feedback: bool = False,
    spam_threshold: float = DEFAULT_SPAM_THRESHOLD,
) -> list[BattleResult]:
    """Train candidate rounds on labeled simulations and score a fixed holdout."""
    if rounds < 1 or examples_per_round < 1:
        raise ValueError("Rounds and examples per round must be positive")

    train_scams, holdout_scams, train_legitimate, holdout_legitimate = split_self_play_examples(
        seed, holdout_per_class
    )
    if rounds * examples_per_round > min(len(train_scams), len(train_legitimate)):
        raise ValueError("Training rounds exceed the available generated examples")

    baseline_model = joblib.load(source_model_path)
    history = [evaluate_messages(baseline_model, holdout_scams, holdout_legitimate, spam_threshold)]

    base_texts, base_labels = load_training_messages(
        data_path,
        smishing_data_path,
        mendeley_data_path,
        nus_reviewed_data_path,
        multilingual_data_path,
    )
    if include_reviewed_feedback:
        feedback_texts, feedback_labels = load_reviewed_feedback(feedback_path)
        base_texts.extend(feedback_texts)
        base_labels.extend(feedback_labels)

    for round_index in range(rounds):
        start = round_index * examples_per_round
        end = start + examples_per_round
        round_scams = train_scams[start:end]
        round_legitimate = train_legitimate[start:end]
        training_texts = base_texts + train_scams[:end] + train_legitimate[:end]
        training_labels = base_labels + ["spam"] * end + ["ham"] * end

        candidate_model = create_pipeline()
        candidate_model.fit(training_texts, training_labels)
        history.append(
            evaluate_messages(candidate_model, holdout_scams, holdout_legitimate, spam_threshold)
        )

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate_model, output_model_path)
    return history


def print_result(label: str, result: BattleResult) -> None:
    print(f"{label}: caught {result.caught_scams}/{result.examples_per_side} scams; "
          f"missed {len(result.missed_scams)}; "
          f"false alarms {len(result.false_alarms)}/{result.examples_per_side}; "
          f"balanced score {result.balanced_score:.1%}")


def print_examples(title: str, messages: tuple[str, ...]) -> None:
    if messages:
        print(f"\n{title}:")
        for message in messages[:5]:
            print(f"- {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--examples", type=int, default=25, help="examples per side (1-100)")
    parser.add_argument("--seed", type=int, default=42, help="seed for repeatable battles")
    parser.add_argument("--spam-threshold", type=float, default=DEFAULT_SPAM_THRESHOLD)
    parser.add_argument("--self-train", action="store_true", help="train a candidate model in battle rounds")
    parser.add_argument("--rounds", type=int, default=3, help="self-training rounds (default: 3)")
    parser.add_argument("--examples-per-round", type=int, default=20)
    parser.add_argument("--holdout-per-class", type=int, default=20)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--smishing-data", type=Path, default=DEFAULT_SMISHING_DATA)
    parser.add_argument("--mendeley-data", type=Path, default=DEFAULT_MENDELEY_DATA)
    parser.add_argument("--nus-reviewed-data", type=Path, default=DEFAULT_NUS_REVIEWED_DATA)
    parser.add_argument("--multilingual-data", type=Path, default=DEFAULT_MULTILINGUAL_DATA)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--include-reviewed-feedback", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE_MODEL)
    args = parser.parse_args()

    try:
        if args.self_train:
            history = train_battle_candidate(
                source_model_path=Path(args.model),
                data_path=args.data,
                smishing_data_path=args.smishing_data,
                mendeley_data_path=args.mendeley_data,
                nus_reviewed_data_path=args.nus_reviewed_data,
                multilingual_data_path=args.multilingual_data,
                output_model_path=args.output,
                feedback_path=args.feedback,
                rounds=args.rounds,
                examples_per_round=args.examples_per_round,
                holdout_per_class=args.holdout_per_class,
                seed=args.seed,
                include_reviewed_feedback=args.include_reviewed_feedback,
                spam_threshold=args.spam_threshold,
            )
            print("Synthetic battle training uses simulator labels, never the model's predictions.")
            print_result("Baseline", history[0])
            for round_number, result in enumerate(history[1:], start=1):
                print_result(f"Round {round_number}", result)
            print(f"Candidate model saved to {args.output}")
            print("The active model was not replaced. Review the results before promoting this candidate.")
            return 0

        model = joblib.load(args.model)
        result = evaluate_battle(model, args.examples, args.seed, args.spam_threshold)
    except (OSError, ValueError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Synthetic message battle: local simulator vs. SMS spam classifier")
    print_result("Defender", result)
    print_examples("Missed simulated scams", result.missed_scams)
    print_examples("Legitimate examples flagged as spam", result.false_alarms)
    print("\nThis synthetic challenge is not a measure of real-world fraud detection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())