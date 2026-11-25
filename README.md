# CardHistory-Fraud: card fraud that hides in the history

Code for the CardHistory-Fraud benchmark: the generator that builds the dataset from IBM's TabFormer card transactions, the
evaluation script, and every baseline in the table below (transformer weights with training and scoring code, and the
XGBoost baseline).

- Dataset, dataset card and baseline files: https://huggingface.co/datasets/toloka/CardHistory-Fraud

The benchmark is 24.9 million card transactions (2,000 cardholders, 6,139 cards, 1991–2020): IBM's simulated transactions plus
fraud that can only be found in a card's history (card testing, stolen and counterfeit cards, breached merchants, testing waves,
fraudulent recurring charges), legit look-alikes (trips, new subscriptions) and a bank that blocks cards after fraud. The dataset
card describes what is inside.

## Install

```bash
pip install -r requirements.txt        # Python 3.13; numpy, polars, pyarrow, pandas, scikit-learn, torch, xgboost
```

## Evaluate a model

Download the dataset repository from https://huggingface.co/datasets/toloka/CardHistory-Fraud into a folder `dataset/` (it holds `data/part-00.parquet` … `part-12.parquet`
and the baseline files in `baselines/`), train on the rows with
`split == "train"`, pick your model on `split == "validation"`, and score every test transaction. A scores file is a parquet
(or CSV) file with two columns: `row_id` and `score` (higher = more suspicious), one row per test transaction.

```bash
python evaluate.py --data dataset/ --scores run1.parquet run2.parquet run3.parquet \
                   --dev-scores run1_val.parquet run2_val.parquet run3_val.parquet
```

The benchmark's result is **fraud caught**: the share of the test period's fraud a model catches when its alerts hold at most 5
legit transactions per caught fraud. To get it, `evaluate.py` sorts the test transactions by the model's score, highest first,
and alerts on the longest top of the list that holds at most 5 legit transactions per caught fraud (equal scores are never
split). IBM's original fraud is taken out of the list first: it counts neither as fraud nor as a false alarm. (It is training
data like any other fraud: our models learned from it.) `evaluate.py` prints five views:

| View | Fraud that counts |
|---|---|
| headline | the fraud we added, without the attempts on a blocked card; IBM's original fraud left out |
| without probes | the headline without the small test payments of distributed testing and testing waves |
| with blocked-card attempts | the headline plus the fraudster's attempts on a blocked card |
| rarer fraud | the headline at 2 false alarms per caught fraud |
| alert line fixed in advance | the alert line set on the validation set (`--dev-scores`), applied to the test period |

## Baselines

Fraud caught (%), test period; transformers: mean of 3 training runs (seeds 0, 1, 2).

| Model | Headline | Without probes | With blocked-card attempts | Rarer fraud | Alert line fixed in advance |
|---|---:|---:|---:|---:|---:|
| Teacher 5M | 85.0 | 75.3 | 93.6 | 78.8 | 84.1 |
| Student 2.6M, distilled from Teacher 5M | 85.0 | 75.0 | 93.7 | 78.8 | 84.2 |
| Student 0.4M, distilled from Teacher 5M | 83.6 | 73.1 | 93.0 | 77.1 | 82.7 |
| Student 0.4M, trained on the labels alone | 79.7 | 67.1 | 91.1 | 73.2 | 78.1 |
| XGBoost with 46 hand-made features | 76.1 | 60.4 | 90.7 | 68.2 | 74.7 |

| Model | Architecture | Parameters (without / with embedding tables) | CPU time per transaction (1 thread) |
|---|---|---|---|
| Teacher 5M | 6 layers, width 256, 8 heads | 5.0M / 6.1M | 15.5 ms |
| Student 2.6M | Teacher 5M with its layers 0, 1 and 5, then distilled | 2.6M / 3.7M | 8.8 ms |
| Student 0.4M | 3 layers, width 96, 4 heads | 0.4M / 1.5M | 3.8 ms |

All three read a card's last 128 transactions as they are (amount, chip / swipe / online, MCC, merchant, city, state, decline
code, hour, day of week, time since the previous transaction), with no hand-made features; each field has its own embedding
table. The students learn from Teacher 5M seed 0. CPU time: median over 100 single-window calls, preprocessing not included.

The baseline files are in the dataset repository (https://huggingface.co/datasets/toloka/CardHistory-Fraud), folder `baselines/`:
- `scores/test/*.parquet`, `scores/validation/*.parquet`: the scores behind the table, one file per model and seed;
- `weights/*.safetensors`: the 12 transformers (configuration and training recipe in each file's metadata; index in `models.json`);
- `weights/xgboost-geo.json`: the XGBoost model.

## Reproduce the table

**From the published scores** (gives the table exactly):

```bash
for m in teacher-5m student-2.6m-distilled student-0.4m-distilled student-0.4m-labels-only; do
  python evaluate.py --data dataset/ --scores dataset/baselines/scores/test/$m-seed{0,1,2}.parquet \
                     --dev-scores dataset/baselines/scores/validation/$m-seed{0,1,2}.parquet
done
python evaluate.py --data dataset/ --scores dataset/baselines/scores/test/xgboost-geo.parquet --dev-scores dataset/baselines/scores/validation/xgboost-geo.parquet
```

**From the released weights:**

```bash
python models/prepare.py --data dataset/ --out arrays/                      # about 1 minute: model inputs for every transaction
python models/score.py --arrays arrays/ --model dataset/baselines/weights/student-0.4m-distilled-seed0.safetensors \
                       --split test --out scores/test/student-0.4m-distilled-seed0.parquet
python models/score.py --arrays arrays/ --model dataset/baselines/weights/student-0.4m-distilled-seed0.safetensors \
                       --split validation --out scores/validation/student-0.4m-distilled-seed0.parquet
python baselines/xgboost/predict.py --arrays arrays/ --model dataset/baselines/weights/xgboost-geo.json --split test --out scores/test/xgboost-geo.parquet
```

The published scores were made on a GPU in bfloat16; on a CPU in float32 the scores differ slightly. On our check (CPU, 2 threads) the
re-scored logits stayed within 0.24 of the published ones (correlation 0.9999 for all 12 models), and on the full test set the
recalls moved by at most 0.3 points (3 to 13 of the 4,952 headline frauds changed between caught and missed). Scoring the test
and validation sets took 3 minutes for Student 0.4M, 8 for Student 2.6M and 13 for Teacher 5M.

How a transformer scores a transaction: each card's history is cut into windows of 128 transactions that overlap by half, and
each transaction takes its score from the first window in which it is one of the last 64. So every score sees at least 64 earlier
transactions of the card (or all of them, for a card's first transactions), and no score depends on batching.

**Retrain everything** (GPU):

```bash
scripts/reproduce.sh dataset/ work/
```

It prepares the arrays, trains Teacher 5M (3 seeds), the three students (3 seeds each, from the retrained Teacher 5M seed 0) and
XGBoost, scores the test and validation sets and prints the table. The recipes are in `models/recipes.json`; in our run
each transformer trained in 9 to 29 minutes, with up to 5 trainings sharing one GPU. Retrained models are new models: their
numbers will be close to the table, not equal to it. `scripts/smoke_test.sh` runs every step on a CPU in a few minutes (a few
training steps per model; the numbers mean nothing) to check an installation.

## Build the dataset yourself

```bash
python generator/make_dataset.py --ibm-csv card_transaction.v1.csv --out dataset/
```

`card_transaction.v1.csv` is in `data/credit_card/transactions.tgz` of IBM's repository (https://github.com/IBM/TabFormer).
The build takes about 10 minutes and under 8 GB of memory. With the versions in `requirements.txt` it gives exactly the published
dataset: the `sha256_core_arrays` of your `FINGERPRINT.json` equal the published ones. Other numpy versions may draw different
random numbers. Every setting of the generator is listed, with what it means, in `PARAMS` and `BANK` in `generator/build.py`.

## What we checked before release

- The released generator rebuilds the published dataset, row for row in all columns, from IBM's file.
- `models/prepare.py` gives the arrays the models were trained on, bit for bit (token ids, numeric inputs, labels).
- `evaluate.py` on the published scores gives the table above.
- The released weights are the checkpoints behind the table (sha256 equal to the ones recorded before the test set was scored); scored on a
  CPU with this code, they agree with the published scores to a correlation of 0.9999 (all 12 models; recalls within 0.3 points
  on the full test set).
- `models/train.py` is our original training code reduced to the four recipes: run side by side on a CPU from the same seed, both
  give the same weights bit for bit after the first training steps of every recipe.
- `baselines/xgboost/`: from the release's features, the released model gives the published test scores bit for bit; a retrain
  with `train.py` on a CPU caught 76.3% in the headline view (the released GPU-trained model: 76.1%).

## Repository layout

| Path | What it is |
|---|---|
| `evaluate.py` | the benchmark's metric and its five views |
| `generator/` | `make_dataset.py` (builds the dataset files), `build.py` (the patterns, look-alikes and bank), `city_geo.json` (city coordinates and time zones for travel times) and `build_city_geo.py` (how that table was made) |
| `models/` | `prepare.py` (model inputs), `data.py` (windows over a card's history), `model.py` (the transformer), `train.py` and `recipes.json` (training), `score.py` (scoring), `weights.py` (reading and writing the weights) |
| `baselines/xgboost/` | `features.py` (the 46 features), `train.py`, `predict.py` |
| `scripts/` | `reproduce.sh` (retrain and score everything), `smoke_test.sh` (a quick check on a CPU) |

## Licence

Copyright 2026 Toloka AI BV. The code is released under the Apache License 2.0 (`LICENSE`). The dataset builds on IBM's TabFormer data, which IBM released
under the Apache License 2.0, and is released under the same licence. `NOTICE` lists what we changed in IBM's data and the sources
of the city coordinates (GeoNames, CC BY 4.0). This work is not affiliated with or endorsed by IBM.

## Authors

- Igor Tica: generator, models, experiments
- Oleg Zubkov: project lead, expert review design and analysis, writing

## Citation

```bibtex
@misc{cardhistoryfraud2026,
  title  = {CardHistory-Fraud: card fraud that hides in the history},
  author = {Tica, Igor and Zubkov, Oleg},
  year   = {2026},
  note   = {Toloka},
  url    = {https://huggingface.co/datasets/toloka/CardHistory-Fraud}
}
```

IBM's TabFormer: Padhi et al., Tabular Transformers for Modeling Multivariate Time Series, ICASSP 2021 (arXiv:2011.01843);
Altman, Synthesizing Credit Card Transactions (arXiv:1910.03033).
