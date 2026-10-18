# ScamSheild

## SMS spam screening prototype

This project includes a text classifier trained on the [UCI SMS Spam
Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
(5,574 messages) and the [Smishing Dataset IMC25](https://github.com/reportsmishing/Smishing-Dataset-IMC25)
(33,869 public user reports), plus Mendeley's [SMS Phishing Dataset for
Machine Learning and Pattern Recognition](https://data.mendeley.com/datasets/f45bkkt8pr/1)
(5,971 labeled messages) and the [SMS Scam Detection Dataset](https://github.com/vinit9638/SMS-scam-detection-dataset)
(138,813 labeled entries). The agents also use the published
[Smishing Dataset I paper](https://arxiv.org/abs/2402.18430) as high-level
guidance; its Smishtank report API is not imported into training. The UCI corpus
labels messages `spam` or `ham`; the NUS SMS Corpus is also downloaded as an
unlabeled review queue and contributes only messages explicitly labeled by a
human reviewer.
IMC25 contributes smishing-positive examples and English translations when
available; Mendeley labels examples `ham`, `spam`, or `Smishing`, with spam
and smishing mapped to the model's positive class. The additional dataset has
`ham`/`spam` labels and sparse language annotations. This is still a limited
text classifier, not a general fraud detector; use predictions as screening
signals, not definitive judgments.

### Setup and training

```sh
python -m pip install -r requirements.txt
python sms_spam.py train
```

Training combines the UCI TSV and downloaded labeled CSVs, plus only manually
reviewed NUS records. It removes exact normalized duplicates and excludes
conflicting labels, then adds 200 clearly fictional examples per class.
Evaluation holds out a stratified 20% of the combined corpus before adding
synthetic examples to the training fold. The final model is fit on the
combined corpora and synthetic examples at
`models/sms_spam.joblib`. By default IMC25 is read from
`data/Smishing-Dataset-IMC25/final_dataset_output.csv`; override it with
`--smishing-data`. Mendeley is read from
`data/Mendeley-SMS-Phishing/Dataset_5971.csv`; override it with
`--mendeley-data`.
The multilingual ham/spam CSV is at
`data/SMS-Scam-Detection-Multilingual/sms_scam_detection_dataset_merged_with_lang.csv`;
override it with `--multilingual-data`.

The NUS source archive is under `data/NUS-SMS-Corpus/`; its 55,835 English
messages are listed in `review_queue.csv`. The corpus has no fraud/ham labels.
After reviewing a record, enter `ham` or `spam` in its `reviewed_label` column.
Leave all uncertain or unreviewed rows blank. Run the normal training command
again to include only reviewed labels. These messages may contain personal
content; keep the review queue private.

### Classify a message

```sh
python sms_spam.py predict "Your account is locked. Click this link to verify."
```

The output includes the predicted `spam`/`ham` label and the model's spam
probability. This probability is not a guarantee of fraud.

### Run a synthetic detector battle

#### `fraud_simulator.py`

This module builds a fixed library of fictional scam-like and ordinary SMS
messages from invented organization names, event descriptions, and reserved
`.invalid` domains. It provides deterministic sampling functions for both
classes, so a given count and seed returns a repeatable batch. The examples
are synthetic test inputs, not copied dataset records, real reports, or
messages to send. The training script also uses 200 generated examples from
each class as supplementary labeled training data.

#### `fraud_battle.py`

This command evaluates the saved local SMS classifier against an equal-sized
batch from the simulator. It reports scam detection rate, missed scam examples,
false alarms on legitimate examples, and a balanced score that averages scam
recall with legitimate-message specificity. The default battle is 25 examples
per class; `--examples` and `--seed` control batch size and reproducibility.
These scores measure performance on this simulator only, not real-world fraud
detection or the Copilot custom agents.

With `--self-train`, the command trains successive candidate models by adding
known simulator labels, evaluates them on a separate synthetic holdout, and
saves the final candidate to `models/sms_spam_battle_candidate.joblib` (or the
path given by `--output`). It does not use the model's own predictions as
labels and does not replace the active model. Review the candidate and its
results separately before choosing to promote it.

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

The web app and battle use a 45% spam-score cutoff by default. The latest
combined-corpus held-out run reported 84% accuracy, 0.85 spam recall, and 0.82
ham recall. These scores depend on the dataset split and do not establish
real-world fraud performance. Use
`--spam-threshold 0.5` to change the decision cutoff; a lower cutoff usually
catches more borderline spam but can also flag more legitimate messages.

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
Use the header control to switch between light and dark appearance; the choice
is saved in this browser.
Smishing means phishing delivered by SMS. The training data includes smishing
examples, which the binary classifier groups into its `spam` class rather than
reporting as a separate category.

Use **Generate fraudulent sample** or **Generate non-fraudulent sample** for a
fictional SMS written in a compact, varied style informed by broad patterns in
the SMS corpora; examples are newly written and do not copy their messages.
The generator uses invented service names and reserved, non-routable
`.invalid` links rather than bracketed placeholders. The generated class label
and the model's independent text-based decision are shown separately, along
with whether they agree. Generated labels are test ground truth only and are
not supplied to the classifier. Synthetic agreement is not real-world
accuracy. The interface does not ask users to label or rate messages and does
not retrain the classifier from screening sessions; model training is performed
separately with `python sms_spam.py train`.

### Data source

The UCI corpus file at `data/SMSSpamCollection` is downloaded unchanged from
UCI; see its dataset page for source information and applicable terms.

The IMC25 CSV and its `LICENSE.txt` are kept under
`data/Smishing-Dataset-IMC25/`. The dataset is provided under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Sharad Agarwal,
Antonis Papasavva, Guillermo Suarez-Tangil, and Marie Vasek. Cite the work:

> Sharad Agarwal, Antonis Papasavva, Guillermo Suarez-Tangil, and Marie Vasek.
> “Fishing for Smishing: Understanding SMS Phishing Infrastructure and
> Strategies by Mining Public User Reports.” Proceedings of the 2025 ACM on
> Internet Measurement Conference (IMC '25), 2025.
> [https://doi.org/10.1145/3730567.3764431](https://doi.org/10.1145/3730567.3764431)

The Mendeley CSV is kept at `data/Mendeley-SMS-Phishing/Dataset_5971.csv`.
It is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
by Sandhya Mishra and Devpriya Soni (2022), DOI
[10.17632/f45bkkt8pr.1](https://doi.org/10.17632/f45bkkt8pr.1), titled
“SMS Phishing Dataset for Machine Learning and Pattern Recognition.”

The multilingual SMS Scam Detection CSV is kept under
`data/SMS-Scam-Detection-Multilingual/` and was downloaded from
[vinit9638/SMS-scam-detection-dataset](https://github.com/vinit9638/SMS-scam-detection-dataset).
The repository describes 138,813 ham/spam entries; language labels are
available for only a subset. The repository has no declared license file; its
reuse was authorized by the project owner for this integration. Do not infer
language coverage from the repository's stated list alone.

The English NUS SMS Corpus is downloaded from
[kite1988/nus-sms-corpus](https://github.com/kite1988/nus-sms-corpus), release
March 9, 2015. Its README requests citation of Tao Chen and Min-Yen Kan (2013),
“Creating a Live, Public Short Message Service Corpus: The NUS SMS Corpus,”
*Language Resources and Evaluation*, 47(2), 299–355, and asks users planning
to use the corpus to notify Tao Chen. The repository does not provide an explicit license file;
use the corpus only under the permission confirmed by the user. We cannot send
the requested notice on the user's behalf.

The [Smishing Dataset I paper](https://arxiv.org/abs/2402.18430) describes
community-submitted reports and associated sender, brand, and URL/domain
metadata. The paper points to Smishtank for ongoing public submissions, but
this project does not download or retain those reports: the API did not return
a usable report list during integration checks, and no explicit reuse license
for the submitted messages was identified. The agents use only the paper's
high-level descriptions and do not claim live API access.

The dataset contains message text, translations, URLs, and sender-related
metadata. The Mendeley dataset also contains URL/email/phone indicators. Do not
visit dataset URLs or copy message text, phone numbers, or other details into
simulator examples. The custom agents use the datasets as high-level context;
their instructions are not model training.

### Works Cited (MLA)

Agarwal, Sharad, et al. “Fishing for Smishing: Understanding SMS Phishing
Infrastructure and Strategies by Mining Public User Reports.” *Proceedings of
the 2025 ACM on Internet Measurement Conference*, Association for Computing
Machinery, 2025, https://doi.org/10.1145/3730567.3764431. CC BY 4.0.

Almeida, Tiago A., et al. “Contributions to the Study of SMS Spam Filtering:
New Collection and Results.” *Proceedings of the 2011 ACM Symposium on
Document Engineering*, Association for Computing Machinery, 2011, pp. 259–62,
https://doi.org/10.1145/2034691.2034742.

Chen, Tao, and Min-Yen Kan. “Creating a Live, Public Short Message Service
Corpus: The NUS SMS Corpus.” *Language Resources and Evaluation*, vol. 47,
no. 2, 2013, pp. 299–355, https://doi.org/10.1007/s10579-012-9197-9.

Kite1988. “NUS SMS Corpus.” *GitHub*, 9 Mar. 2015,
https://github.com/kite1988/nus-sms-corpus. Accessed 18 Oct. 2026.

Mishra, Sandhya, and Devpriya Soni. “SMS Phishing Dataset for Machine Learning
and Pattern Recognition.” *Mendeley Data*, 2022,
https://doi.org/10.17632/f45bkkt8pr.1. CC BY 4.0.

“SMS-scam-detection-dataset.” *GitHub*,
https://github.com/vinit9638/SMS-scam-detection-dataset. Accessed 18 Oct. 2026.
No license is declared by the repository; reuse for this project was
authorized by its owner.

“SMS Spam Collection.” *UCI Machine Learning Repository*, University of
California, Irvine, https://archive.ics.uci.edu/dataset/228/sms+spam+collection.
Accessed 18 Oct. 2026.

Timko, Daniel, and Muhammad Lutfor Rahman. “Smishing Dataset I: Phishing SMS
Dataset from Smishtank.com.” *arXiv*, 28 Feb. 2024,
https://doi.org/10.1145/3626232.3653282. Cited for background only; its
Smishtank messages are not downloaded or included in model training.