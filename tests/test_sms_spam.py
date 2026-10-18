import json
import tempfile
import unittest
import warnings
from pathlib import Path

from app import create_app
from fraud_battle import evaluate_battle, split_self_play_examples
from fraud_simulator import generate_fraud_messages, generate_legitimate_messages
from sms_spam import (
    load_messages,
    load_mendeley_messages,
    load_multilingual_messages,
    load_reviewed_feedback,
    load_reviewed_nus_messages,
    load_smishing_messages,
    load_training_messages,
    load_user_feedback,
    predict_messages,
    train,
)


class LoadMessagesTests(unittest.TestCase):
    def test_loads_labels_and_preserves_tabs_in_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "messages.tsv"
            dataset.write_text("ham\tHello\tthere\nspam\tClaim your prize\n", encoding="utf-8")

            texts, labels = load_messages(dataset)

        self.assertEqual(texts, ["Hello\tthere", "Claim your prize"])
        self.assertEqual(labels, ["ham", "spam"])

    def test_loads_smishing_csv_and_prefers_translation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "smishing.csv"
            dataset.write_text(
                "text,translation,scam_type\n"
                '"Hola","Hello","banking"\n'
                '"Original English","","delivery"\n',
                encoding="utf-8",
            )

            texts, labels = load_smishing_messages(dataset)

        self.assertEqual(texts, ["Hello", "Original English"])
        self.assertEqual(labels, ["spam", "spam"])

    def test_loads_mendeley_csv_and_maps_spam_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "mendeley.csv"
            dataset.write_text(
                "LABEL,TEXT,URL,EMAIL,PHONE\n"
                'ham,"A normal message",No,No,No\n'
                'Spam,"An unwanted offer",Yes,No,No\n'
                'smishing,"A suspicious notice",Yes,No,No\n',
                encoding="utf-8",
            )

            texts, labels = load_mendeley_messages(dataset)

        self.assertEqual(texts, ["A normal message", "An unwanted offer", "A suspicious notice"])
        self.assertEqual(labels, ["ham", "spam", "spam"])

    def test_loads_multilingual_ham_spam_labels_and_ignores_sparse_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "multilingual.csv"
            dataset.write_text(
                "label,text,lang,URL,,,,\n"
                'ham,"A normal message",en,No,,,\n'
                'SPAM,"An unwanted message",fr,Yes,,,\n',
                encoding="utf-8",
            )

            texts, labels = load_multilingual_messages(dataset)

        self.assertEqual(texts, ["A normal message", "An unwanted message"])
        self.assertEqual(labels, ["ham", "spam"])

    def test_rejects_unknown_multilingual_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "multilingual.csv"
            dataset.write_text(
                "label,text\n"
                'unsure,"An uncertain message"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid label"):
                load_multilingual_messages(dataset)

    def test_warns_and_skips_empty_multilingual_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "multilingual.csv"
            dataset.write_text(
                "label,text\n"
                'ham,""\n'
                'spam,"A labeled message"\n',
                encoding="utf-8",
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                texts, labels = load_multilingual_messages(dataset)

        self.assertEqual(texts, ["A labeled message"])
        self.assertEqual(labels, ["spam"])
        self.assertEqual(len(caught), 1)
        self.assertIn("Skipping empty message", str(caught[0].message))

    def test_rejects_unknown_mendeley_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "mendeley.csv"
            dataset.write_text(
                "LABEL,TEXT\n"
                'unknown,"A message with an unknown label"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid label"):
                load_mendeley_messages(dataset)

    def test_loads_only_human_reviewed_nus_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "nus-review.csv"
            dataset.write_text(
                "message_id,text,reviewed_label\n"
                '1,"Unreviewed message",\n'
                '2,"Confirmed normal message",ham\n'
                '3,"Confirmed scam message",spam\n',
                encoding="utf-8",
            )

            texts, labels = load_reviewed_nus_messages(dataset)

        self.assertEqual(texts, ["Confirmed normal message", "Confirmed scam message"])
        self.assertEqual(labels, ["ham", "spam"])

    def test_rejects_invalid_human_reviewed_nus_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "nus-review.csv"
            dataset.write_text(
                "message_id,text,reviewed_label\n"
                '1,"Uncertain message",unsure\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid reviewed label"):
                load_reviewed_nus_messages(dataset)

    def test_combined_training_data_deduplicates_and_drops_conflicting_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ham = root / "ham.tsv"
            smishing = root / "smishing.csv"
            ham.write_text("ham\tA legitimate note\nspam\tAn existing spam\n", encoding="utf-8")
            smishing.write_text(
                "text,translation\n"
                '"A LEGITIMATE   note",\n'
                '"An additional smishing message",\n'
                '"An existing spam",\n',
                encoding="utf-8",
            )

            texts, labels = load_training_messages(ham, smishing, None, None, None)

        self.assertEqual(
            dict(zip(texts, labels)),
            {
                "An additional smishing message": "spam",
                "An existing spam": "spam",
            },
        )

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
        self.client = create_app(
            model=FakeModel(),
        ).test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_home_page_exposes_accessible_theme_toggle(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="theme-toggle"', response.data)
        self.assertIn(b"Switch to dark mode", response.data)
        self.assertIn(b'aria-pressed="false"', response.data)

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

    def test_generates_distinct_non_deployable_examples(self) -> None:
        first_response = self.client.post(
            "/api/generate-sample", json={"sample_type": "fraud"}
        )
        first = first_response.get_json()["message"]
        second_response = self.client.post(
            "/api/generate-sample", json={"sample_type": "fraud"}
        ).get_json()

        self.assertNotEqual(first, second_response["message"])
        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(
            first_response.get_json()["notice"],
            "SIMULATED - DO NOT SEND. Fictional sample; links use the reserved .invalid domain.",
        )
        self.assertEqual(first_response.get_json()["ground_truth"], "fraudulent")
        self.assertEqual(first_response.get_json()["detector_verdict"], "non-fraudulent")
        self.assertFalse(first_response.get_json()["matches_ground_truth"])
        self.assertNotIn("[", first)
        self.assertNotIn("]", first)
        self.assertIn(".invalid", first)
        self.assertNotIn("SIMULATED - DO NOT SEND", first)

    def test_generates_legitimate_sample_and_reports_detector_agreement(self) -> None:
        response = self.client.post(
            "/api/generate-sample", json={"sample_type": "legitimate"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["ground_truth"], "non-fraudulent")
        self.assertEqual(response.get_json()["detector_verdict"], "non-fraudulent")
        self.assertTrue(response.get_json()["matches_ground_truth"])

    def test_rejects_unknown_generated_sample_type(self) -> None:
        response = self.client.post(
            "/api/generate-sample", json={"sample_type": "unknown"}
        )

        self.assertEqual(response.status_code, 400)

    def test_feedback_endpoint_is_removed(self) -> None:
        response = self.client.post(
            "/api/feedback",
            json={"actual_status": "fraudulent"},
        )

        self.assertEqual(response.status_code, 404)


class FraudBattleTests(unittest.TestCase):
    def test_simulator_generates_reproducible_balanced_batches(self) -> None:
        scams = generate_fraud_messages(8, seed=7)
        repeated_scams = generate_fraud_messages(8, seed=7)
        legitimate = generate_legitimate_messages(8, seed=8)

        self.assertEqual(scams, repeated_scams)
        self.assertEqual(len(set(scams)), 8)
        self.assertEqual(len(legitimate), 8)
        self.assertTrue(all("[" not in message and "]" not in message for message in scams + legitimate))
        self.assertTrue(all(".invalid" in message for message in scams))

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
            smishing_path = Path(directory) / "smishing.csv"
            mendeley_path = Path(directory) / "mendeley.csv"
            nus_path = Path(directory) / "nus-review.csv"
            rows = [f"ham\tHello friend, how are you today {index}\n" for index in range(12)]
            rows.extend(
                f"spam\tClaim a free prize now at the safe test link {index}\n"
                for index in range(12)
            )
            data_path.write_text("".join(rows), encoding="utf-8")
            smishing_path.write_text(
                "text,translation\n"
                '"A fictional smishing notice",\n'
                '"Another fictional smishing notice",\n',
                encoding="utf-8",
            )
            mendeley_path.write_text(
                "LABEL,TEXT\n"
                'ham,"A fictional legitimate note"\n'
                'smishing,"A fictional smishing note"\n',
                encoding="utf-8",
            )
            nus_path.write_text(
                "message_id,text,reviewed_label\n"
                '1,"A manually reviewed legitimate SMS",ham\n',
                encoding="utf-8",
            )
            feedback_path.write_text(
                json.dumps({
                    "message": "A reviewed fictional warning [SAFE TEST LINK]",
                    "user_label": "fraud",
                    "approved": True,
                }) + "\n",
                encoding="utf-8",
            )

            train(
                data_path,
                model_path,
                feedback_path,
                include_reviewed_feedback=True,
                smishing_data_path=smishing_path,
                mendeley_data_path=mendeley_path,
                nus_reviewed_data_path=nus_path,
                multilingual_data_path=None,
            )

            self.assertTrue(model_path.is_file())


if __name__ == "__main__":
    unittest.main()
