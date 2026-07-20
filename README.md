# IPL Win Predictor

An end-to-end machine learning project that estimates a team's probability of winning an IPL match in real time, based on the current state of play. Rather than just predicting a final winner, it recalculates the win probability after every ball using score, wickets, overs, target, and run rate.

**Live demo:** (add after deployment)

## Overview

The chance of a team winning a cricket match shifts constantly — a wicket, a boundary, or a maiden over can swing things noticeably. A single prediction made before the match starts doesn't capture any of that. This project trains a model on ball-by-ball data so it can produce an updated probability at any point in the innings, based on:

- Current score and target
- Overs completed and wickets remaining
- Current run rate and required run rate
- Batting team, bowling team, and venue

## Dataset

Built on historical IPL ball-by-ball data: `deliveries.csv` and `matches.csv`. Between them they cover match metadata, every delivery bowled, runs scored, wickets, venue, teams, and final results.

## Feature engineering

Raw ball-by-ball data isn't directly useful for this — the model needs match-state features that actually change meaning as the game progresses:

- Runs left
- Balls left
- Wickets remaining
- Current run rate (CRR)
- Required run rate (RRR)
- Target score
- Batting team, bowling team, host city

These were engineered from the raw deliveries data and gave a noticeably better signal than feeding the model raw columns.

## Pipeline

The preprocessing and training pipeline is built with scikit-learn's `Pipeline` and `ColumnTransformer`: categorical fields (teams, city) go through one-hot encoding, and the encoded features feed into a logistic regression classifier that outputs a win probability rather than just a label.

## Model choice: logistic regression over Random Forest

Two models were tested. A Random Forest classifier reached close to 99% accuracy, but its output probabilities were a problem in practice — it would frequently show something like 98% to 2% even in matches that were still genuinely close. High accuracy, badly calibrated probabilities: fine for picking a winner, unusable for a live win-probability display.

Logistic regression came in lower on accuracy, at 82.37%, but its probability estimates were far more realistic. They moved gradually as the match state changed instead of jumping to near-certainty after a single event, which matters a lot more than raw accuracy for a tool meant to be watched ball by ball. That's why logistic regression is the model actually deployed.

| Metric | Value |
|---|---|
| Model | Logistic Regression |
| Accuracy | 82.37% |
| Encoding | OneHotEncoder |
| Pipeline | scikit-learn Pipeline |
| Probability output | Yes |
| Deployment | Streamlit |

## Tech stack

Python, Pandas, NumPy, scikit-learn, Streamlit, Pickle, Git, GitHub.

## Features

- Live win probability that updates with match state
- Interactive Streamlit UI with team and city selection
- Manual input of current match situation
- Probability visualization
- scikit-learn pipeline handling preprocessing end to end

## Challenges

**Feature engineering.** The raw dataset alone wasn't predictive enough. Deriving match-state features like runs left, balls left, and required run rate made the biggest difference in prediction quality.

**Choosing between accuracy and calibration.** Random Forest's near-99% accuracy was tempting, but its overconfident probabilities made it a poor fit for a tool meant to show believable in-match odds. Prioritizing calibration over raw accuracy meant going with the weaker-on-paper logistic regression model instead.

**Deployment friction.** Getting the app running on Streamlit surfaced a handful of version-compatibility issues — mismatches between scikit-learn versions, the pickled model, and a deprecated OneHotEncoder API, along with the usual Python version differences.

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

## Results

The deployed model produces win probabilities that shift realistically as a match unfolds, which makes it more useful for live match tracking than a model optimized purely for classification accuracy.

## Future improvements

- Incorporate player statistics and toss impact
- Factor in weather conditions
- Try XGBoost or LightGBM with explicit probability calibration
- Explore sequence models for ball-by-ball prediction
- Integrate a live data feed (Cricbuzz / CricAPI) for real-time predictions during a match

## Author

**Prince Gupta** — B.Tech CSE, Swami Keshvanand Institute of Technology (SKIT)

GitHub: https://github.com/prince8094
LinkedIn: https://www.linkedin.com/in/prince-gupta-8a285a328/
