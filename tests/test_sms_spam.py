import json
import tempfile
import unittest
from pathlib import Path

import joblib

from app import create_app
from fraud_battle import evaluate_battle, split_self_play_examples
from fraud_simulator import generate_fraud_messages, generate_legitimate_messages
from sms_spam import load_messages, load_reviewed_feedback, load_user_feedback, predict_messages, train


class LoadMessagesTests(unittest.TestCase):
    def test_loads_labels_and_preserves_tabs_in_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "messages.tsv"
            dataset.write_text("ham\tHello\tthere\nspam\tClaim your prize\n", encoding="utf-8")

            texts, labels = load_messages(dataset)

        self.assertEqual(texts, ["Hello\tthere", "Claim your prize"])
        self.assertEqual(labels, ["ham", "spam"])

    def test_rejects_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "messages.tsv"
            dataset.write_text("unknown\tHello\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid label"):
                load_messages(dataset)

    def test_loads_only_approved_feedback_as_spam_and_ham(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feedback = Path(directory) / "feedback.jsonl"
            feedback.write_text(
                '\n'.join([
                    json.dumps({"message": "Pending", "user_label": "fraud", "approved": False}),
                    json.dumps({"message": "Approved scam", "user_label": "fraud", "approved": True}),
                    json.dumps({"message": "Approved real", "user_label": "legitimate", "approved": True}),
                ]),
                encoding="utf-8",
            )

            texts, labels = load_reviewed_feedback(feedback)

        self.assertEqual(texts, ["Approved scam", "Approved real"])
        self.assertEqual(labels, ["spam", "ham"])

    def test_skips_legacy_feedback_without_message_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feedback = Path(directory) / "feedback.jsonl"
            feedback.write_text(
                '\n'.join([
                    json.dumps({"timestamp": "old", "user_label": "fraud"}),
                    json.dumps({"message": "New feedback", "user_label": "fraud", "approved": True}),
                ]),
                encoding="utf-8",
            )

            texts, labels = load_user_feedback(feedback)

        self.assertEqual(texts, ["New feedback"])
        self.assertEqual(labels, ["spam"])


class FakeModel:
    classes_ = ["ham", "spam"]

    def predict(self, messages: list[str]) -> list[str]:
        return ["spam" if "claim" in message.lower() else "ham" for message in messages]

    def predict_proba(self, messages: list[str]) -> list[list[float]]:
        prediction = self.predict(messages)[0]
        return [[0.1, 0.9] if prediction == "spam" else [0.9, 0.1]]


class AlwaysHamModel:
    classes_ = ["ham", "spam"]

    def predict(self, messages: list[str]) -> list[str]:
        return ["ham"] * len(messages)

    def predict_proba(self, messages: list[str]) -> list[list[float]]:
        return [[0.6, 0.4] for _ in messages]


class BoundaryScoreModel:
    classes_ = ["ham", "spam"]

    def __init__(self, spam_score: float) -> None:
        self.spam_score = spam_score

    def predict_proba(self, messages: list[str]) -> list[list[float]]:
        return [[1 - self.spam_score, self.spam_score] for _ in messages]


class AnalyzeMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.feedback_path = root / "feedback.jsonl"
        self.model_path = root / "model.joblib"
        self.data_path = root / "messages.tsv"
        rows = [f"ham\tHello friend how are you today {index}\n" for index in range(12)]
        rows.extend(f"spam\tClaim a free prize now today {index}\n" for index in range(12))
        self.data_path.write_text("".join(rows), encoding="utf-8")
        self.client = create_app(
            model_path=self.model_path,
            model=FakeModel(),
            feedback_path=self.feedback_path,
            data_path=self.data_path,
        ).test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_returns_model_screening_result(self) -> None:
        response = self.client.post("/api/analyze", json={"message": "Claim your prize"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "prediction": "spam",
                "spam_score": 0.9,
                "spam_threshold": 0.45,
                "verdict": "Likely spam",
            },
        )

    def test_rejects_empty_and_oversized_messages(self) -> None:
        empty_response = self.client.post("/api/analyze", json={"message": "  "})
        long_response = self.client.post("/api/analyze", json={"message": "x" * 5001})

        self.assertEqual(empty_response.status_code, 400)
        self.assertEqual(long_response.status_code, 413)

    def test_accepts_maximum_length_multibyte_message(self) -> None:
        response = self.client.post("/api/analyze", json={"message": "é" * 5000})

        self.assertEqual(response.status_code, 200)

    def test_records_feedback_and_trains_immediately(self) -> None:
        response = self.client.post(
            "/api/feedback",
            json={"actual_status": "fraud", "prediction": "spam", "message": "Private message"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["trained_feedback_examples"], 1)
        self.assertTrue(self.model_path.is_file())
        record = json.loads(self.feedback_path.read_text(encoding="utf-8"))
        self.assertEqual(record["user_label"], "fraud")
        self.assertEqual(record["model_prediction"], "spam")
        self.assertEqual(record["message"], "Private message")
        self.assertTrue(record["approved"])
        followup = self.client.post("/api/analyze", json={"message": "Private message"})
        trained_model = joblib.load(self.model_path)
        expected_predictions, expected_scores = predict_messages(trained_model, ["Private message"])
        self.assertEqual(followup.get_json()["prediction"], expected_predictions[0])
        self.assertEqual(followup.get_json()["spam_score"], expected_scores[0])

    def test_each_feedback_retrains_using_accumulated_labels(self) -> None:
        first = self.client.post(
            "/api/feedback",
            json={"actual_status": "fraud", "prediction": "ham", "message": "A test fraud message"},
        )
        second = self.client.post(
            "/api/feedback",
            json={"actual_status": "legitimate", "prediction": "spam", "message": "A test legitimate message"},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.get_json()["trained_feedback_examples"], 2)
        records = self.feedback_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(records), 2)

    def test_generates_distinct_non_deployable_examples(self) -> None:
        first = self.client.post("/api/generate-sample").get_json()["message"]
        second = self.client.post("/api/generate-sample").get_json()["message"]

        self.assertNotEqual(first, second)
        self.assertIn("SIMULATED - DO NOT SEND", first)
        self.assertIn("[SAFE TEST LINK]", first)

    def test_rejects_invalid_feedback(self) -> None:
        response = self.client.post(
            "/api/feedback",
            json={"actual_status": "unsure", "prediction": "spam"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.feedback_path.exists())


class FraudBattleTests(unittest.TestCase):
    def test_simulator_generates_reproducible_balanced_batches(self) -> None:
        scams = generate_fraud_messages(8, seed=7)
        repeated_scams = generate_fraud_messages(8, seed=7)
        legitimate = generate_legitimate_messages(8, seed=8)

        self.assertEqual(scams, repeated_scams)
        self.assertEqual(len(set(scams)), 8)
        self.assertEqual(len(legitimate), 8)
        self.assertTrue(all("SIMULATED - DO NOT SEND" in message for message in scams + legitimate))

    def test_battle_reports_missed_scams_and_no_false_alarms(self) -> None:
        result = evaluate_battle(AlwaysHamModel(), examples_per_side=6, seed=9)

        self.assertEqual(result.caught_scams, 0)
        self.assertEqual(len(result.missed_scams), 6)
        self.assertEqual(len(result.false_alarms), 0)
        self.assertEqual(result.balanced_score, 0.5)

    def test_spam_cutoff_increases_recall_at_selected_threshold(self) -> None:
        predictions, scores = predict_messages(BoundaryScoreModel(0.45), ["test"])

        self.assertEqual(predictions, ["spam"])
        self.assertEqual(scores, [0.45])

    def test_rejects_invalid_spam_cutoff(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            predict_messages(BoundaryScoreModel(0.45), ["test"], spam_threshold=1.1)

    def test_self_play_holdout_does_not_overlap_training_examples(self) -> None:
        train_scams, holdout_scams, train_legitimate, holdout_legitimate = split_self_play_examples(
            seed=12, holdout_per_class=10
        )

        self.assertFalse(set(train_scams) & set(holdout_scams))
        self.assertFalse(set(train_legitimate) & set(holdout_legitimate))


class TrainReviewedFeedbackTests(unittest.TestCase):
    def test_trains_to_temporary_model_with_approved_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "messages.tsv"
            model_path = Path(directory) / "model.joblib"
            feedback_path = Path(directory) / "feedback.jsonl"
            rows = [f"ham\tHello friend, how are you today {index}\n" for index in range(12)]
            rows.extend(
                f"spam\tClaim a free prize now at the safe test link {index}\n"
                for index in range(12)
            )
            data_path.write_text("".join(rows), encoding="utf-8")
            feedback_path.write_text(
                json.dumps({
                    "message": "A reviewed fictional warning [SAFE TEST LINK]",
                    "user_label": "fraud",
                    "approved": True,
                }) + "\n",
                encoding="utf-8",
            )

            train(data_path, model_path, feedback_path, include_reviewed_feedback=True)

            self.assertTrue(model_path.is_file())


if __name__ == "__main__":
    unittest.main()
