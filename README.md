# ScamSheild

## SMS spam screening prototype

This project includes a text classifier trained on the [UCI SMS Spam
Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
(5,574 messages; dataset DOI: [10.24432/C5CC84](https://doi.org/10.24432/C5CC84)).
The corpus distinguishes `spam` from `ham` (legitimate messages); it was
collected for SMS spam research and is not a general fraud dataset. The model
can miss newer scams and may flag legitimate messages, so use predictions as
screening signals rather than definitive judgments.

### Setup and training

```sh
python -m pip install -r requirements.txt
python sms_spam.py train
```

Training reports precision, recall, and F1 on a stratified held-out split, then
fits the saved model on the full corpus at `models/sms_spam.joblib`.

### Classify a message

```sh
python sms_spam.py predict "Your account is locked. Click this link to verify."
```

The output includes the predicted `spam`/`ham` label and the model's spam
probability. This probability is not a guarantee of fraud.

### Run a synthetic detector battle

```sh
python fraud_battle.py
```

The battle compares the saved SMS classifier against a reproducible, balanced
batch of fictional scam-like and legitimate messages. It reports scam catches,
misses, legitimate messages incorrectly flagged, and the missed examples.
Use `--examples 50` to set the number on each side or `--seed 7` for a
different repeatable batch. This evaluates the local classifier against the
local simulator; it does not run or train the Copilot custom agents and is not
a measure of real-world fraud detection.

To run three rounds of synthetic data augmentation and save a separate
candidate model, use:

```sh
python fraud_battle.py --self-train
```

Each round adds 20 labeled synthetic examples from each class. A disjoint
synthetic holdout scores every round. The current model remains unchanged; the
candidate is saved to `models/sms_spam_battle_candidate.joblib` for review.
This is supervised augmentation from known simulator labels, not learning from
the classifier's guesses or training the Copilot agent.

The ten-round hard-response experiment is saved in
`data/hard_battle_examples.jsonl` with separate train and holdout splits; its
metrics are in `data/hard_battle_results.json`. The resulting candidate model
is `models/sms_spam_hard_candidate.joblib`; it was not promoted to the active
model.

The web app and battle use a 45% spam-score cutoff by default. Five-fold
validation on the UCI corpus found that, with synthetic augmentation, this
raised spam recall from 94.25% to 95.18%, while precision changed from 94.51%
to 91.76% and accuracy from 98.49% to 98.21%. Use `--spam-threshold 0.5` to
restore the standard cutoff; a lower cutoff catches more borderline spam but
also flags more legitimate messages.

The candidate is now promoted at `models/sms_spam.joblib`, so the normal web
app, prediction command, and battle command use it by default. The previous
model is preserved at `models/sms_spam_baseline.joblib`; the original candidate
artifact remains at `models/sms_spam_battle_candidate.joblib`.

### Use the web interface

```sh
python app.py
```

Open <http://127.0.0.1:5000> and paste an SMS to screen it. The interface
uses the same spam/ham model; it does not determine whether a message is
fraudulent, and its score is not a probability of fraud. Keep the model's
limitations in mind and verify unexpected requests independently.

Use **Generate test message** for a new, clearly fictional SMS with placeholder
identities and a nonfunctional link. After screening, users can label the
message as fraudulent or legitimate. Each rating is appended to
`data/screening_feedback.jsonl` and immediately retrains the local classifier
from the UCI corpus, the fixed simulator set, and all accumulated user labels.
The running app switches to the new model as soon as training succeeds. The
feedback file contains full message text; keep it private and do not submit
sensitive messages. Incorrect labels can degrade the classifier. This updates
the SMS classifier, not the Copilot custom agent. Training failures leave the
rating saved for retry with the next feedback submission.

### Data source

The corpus file at `data/SMSSpamCollection` is downloaded unchanged from UCI's
SMS Spam Collection. See UCI's dataset page for its source information and
applicable terms.