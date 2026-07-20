# IPL Win Predictor

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic_Regression-orange?style=flat-square&logo=scikitlearn)

**Live app:** (add after deployment)

## Overview

This is a machine learning app that estimates a team's live win probability during an IPL match, updated ball by ball. Rather than just picking a winner at the end, it tracks how the probability shifts as the match situation changes — current score, wickets in hand, overs bowled, target, run rate, and required run rate.

Win probability in cricket isn't static. It moves with every wicket, boundary, and dot ball, and depends on the interaction between score, wickets remaining, overs left, target, current run rate, required run rate, and the two teams and venue involved. The goal here was to model that continuously rather than just classify the final result.

## Dataset

Built from historical IPL ball-by-ball data — `deliveries.csv` and `matches.csv` — covering match details, every delivery bowled, runs, wickets, venue, teams, and final results.

## Feature engineering

Raw ball-by-ball data isn't directly useful for this, so the following features were derived for each point in the match:

- Runs left
- Balls left
- Wickets remaining
- Current run rate (CRR)
- Required run rate (RRR)
- Target score
- Batting team / bowling team
- Host city

These carry far more predictive signal than the raw columns on their own.

## Pipeline

The full preprocessing and training flow is a single Scikit-Learn pipeline: input data goes through a `ColumnTransformer` with one-hot encoding for categorical fields, then into a logistic regression classifier that outputs a win probability.

## Model selection

Two models were tried.

**Random Forest** hit close to 99% accuracy in testing, but the probabilities it produced weren't usable. It would frequently show something like Team A at 98% and Team B at 2% in situations where the match was still genuinely close. High accuracy, badly calibrated — the kind of overconfidence you don't want in a live prediction tool.

**Logistic Regression** came in lower on raw accuracy but produced probabilities that actually behaved sensibly — they moved gradually as the match developed instead of snapping to extremes after one good over. Since the point of the app is to show a believable probability after every ball, not just to be right at the final whistle, this was the deciding factor. Logistic Regression is the model that shipped.

| Metric | Value |
|---|---|
| Model | Logistic Regression |
| Accuracy | 82.37% |
| Encoding | OneHotEncoder |
| Pipeline | Scikit-Learn Pipeline |
| Probability output | Yes |
| Deployment | Streamlit |

## Tech stack

Python, Pandas, NumPy, Scikit-Learn, Streamlit, Pickle, Git, GitHub.

## Features

- Live win probability estimation
- Team and city selection
- Match situation input (score, overs, wickets, target)
- Probability visualization
- Streamlit UI

## Project structure

```
IPL-Win-Predictor/
├── app.py
├── pipe.pkl
├── requirements.txt
├── analysis.ipynb
└── Datasets/
    ├── deliveries.csv
    └── matches.csv
```

## Installation

```bash
git clone https://github.com/prince8094/IPL-Win-Predictor.git
cd IPL-Win-Predictor
pip install -r requirements.txt
streamlit run app.py
```

## Notable problems along the way

**Choosing accuracy over calibration would have been the easy mistake.** Random Forest looked better on paper at nearly 99% accuracy, but its probability outputs were unreliable for anything live. Prioritizing calibration over raw accuracy is what led to Logistic Regression instead.

**Feature engineering mattered more than model choice.** Runs left, balls left, and the run-rate features made a bigger difference to prediction quality than swapping models did.

**Deployment had the usual friction.** Getting the pickled model to run cleanly on Streamlit meant sorting out scikit-learn version mismatches, a deprecated package, changes to the OneHotEncoder API, and general dependency drift between the training and deployment environments.

## Results

The deployed model produces win probabilities that move realistically as a match unfolds, which matters more for live match analysis than squeezing out a higher classification accuracy would.

## Possible next steps

- Player-level statistics as features
- Toss impact
- Weather conditions
- XGBoost/LightGBM with explicit probability calibration
- Sequence models for ball-by-ball prediction
- Live data via Cricbuzz or CricAPI

## Author

**Prince Gupta** — B.Tech CSE, Swami Keshvanand Institute of Technology (SKIT)

GitHub: https://github.com/prince8094
LinkedIn: https://www.linkedin.com/in/prince-gupta-8a285a328/
