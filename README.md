# ECG Abnormality Classification Under Noise

We are researching how well can deep learning classify different heart conditions from real ECGs of unseen patients, and how its performance holds up against simpler models as the signal gets noisier.

ECG (electrocardiogram) : recording of the heart's electrical activity overtime. Electrodes on the skin pick it up, and each heartbeat produces a specfic wave pattern, which doctors can then read. 

**Team:** Anita Zawadi Lolli Mantero, Kirsten Alessandra Werner, Sara Lorenzo González ·

IE University, Machine Learning & Communication Skills, Fall 2026

## Data
- **PTB-XL v1.0.3** (Wagner et al., 2020), PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/ (licence CC BY 4.0)
- 12-lead, 10 s ECGs at 100 Hz; target = 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP)
- Patient-wise split using the provided `strat_fold`: folds 1–8 train, 9 validation, 10 test
- Data is **not** stored in this repo; download it with the command below.

## How to reproduce
```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data (~0.5 GB, 100 Hz only)
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data/raw/ptb-xl/ --exclude "records500/*"

# 3. Clean and split
python src/data/clean.py
```

## Repository structure
```
src/data/       data cleaning and splitting
src/features/   feature extraction and EDA figures
src/models/     baselines and neural networks
notebooks/      exploration only
reports/        figures and cleaning log
proposal/       LaTeX source of the proposal
```

