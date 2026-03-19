# ⚽ FPL XP Tool — AI-Powered Fantasy Premier League Predictor & Squad Optimizer

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-189AB4?style=for-the-badge)
![PuLP](https://img.shields.io/badge/PuLP-Optimization-2ECC71?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Predict player expected points (xP) using machine learning, then build the mathematically optimal FPL squad — all in one Streamlit dashboard.**

---

## 🎯 What This Tool Does

Most FPL managers rely on gut feel, forums, or outdated spreadsheets. This tool replaces all of that with a data pipeline that:

1. **Fetches live and historical data** from the official FPL API across multiple seasons
2. **Trains an XGBoost model** on real player statistics to predict expected points (xP) for the upcoming gameweek
3. **Integrates odds data** from a betting API to factor in match context (clean sheet probability, goal likelihood)
4. **Solves a linear programming optimization** (via PuLP) to select the highest-xP squad within FPL's budget and position constraints
5. **Presents everything** in an interactive Streamlit dashboard — no code required to use it

---

## 🖥️ Demo

> *Screenshot or GIF of the Streamlit dashboard — add yours here*

```
streamlit run app.py
```

---

## 🏗️ Architecture

```
FPL API (live + historical)
        │
        ▼
  Data Pipeline (pandas)
  ├── Multi-season training data
  ├── Feature engineering
  │     ├── Form, ICT index, xG, xA
  │     ├── Fixture difficulty rating
  │     └── Odds-adjusted match context
        │
        ▼
  XGBoost Regressor
  (predicts xP per player per GW)
        │
        ▼
  PuLP Linear Programme
  (optimizes squad under FPL constraints)
        │
        ▼
  Streamlit Dashboard
  (predictions + optimal squad + transfer suggestions)
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **xP Predictions** | Per-player expected points for any upcoming gameweek |
| 🤖 **XGBoost Model** | Trained on multi-season FPL + understat data |
| 📈 **Odds Integration** | Live match odds inform clean sheet & goal probabilities |
| ⚡ **Squad Optimizer** | PuLP LP solver builds the best 15-man squad within £100m |
| 🔁 **Transfer Planner** | Suggests optimal 1 or 2-transfer moves from your current squad |
| 📱 **Streamlit UI** | No-code dashboard — anyone can use it |
| 🗂️ **Multi-season Data** | Model trained across multiple Premier League seasons for robustness |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Bachu33/FPL-XP-tool.git
cd FPL-XP-tool

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Configuration

Create a `.env` file in the root directory:

```env
ODDS_API_KEY=your_odds_api_key_here   # Get free key at the-odds-api.com
```

> The FPL API is public and requires no authentication.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Data Fetching | `requests`, FPL API |
| Data Processing | `pandas`, `numpy` |
| Machine Learning | `xgboost`, `scikit-learn` |
| Optimization | `PuLP` (Linear Programming) |
| Odds Data | The Odds API |
| Frontend | `Streamlit` |
| Visualization | `matplotlib`, `plotly` |

---

## 📂 Project Structure

```
FPL-XP-tool/
│
├── app.py                  # Main Streamlit application
├── requirements.txt
│
├── data/
│   ├── fetch_fpl_data.py   # FPL API data pipeline
│   ├── fetch_odds.py       # Odds API integration
│   └── historical/         # Cached multi-season datasets
│
├── model/
│   ├── train.py            # XGBoost training pipeline
│   ├── predict.py          # xP prediction for current GW
│   └── xgb_model.pkl       # Saved trained model
│
├── optimizer/
│   └── squad_optimizer.py  # PuLP LP squad optimization
│
└── utils/
    └── helpers.py
```

---

## 🧠 How the Model Works

The XGBoost regressor is trained on features engineered from multiple seasons of FPL and Understat data:

**Input Features:**
- Rolling form (last 3 & 5 GWs)
- ICT index (Influence, Creativity, Threat)
- Expected goals (xG) and expected assists (xA)
- Fixture Difficulty Rating (FDR)
- Home/Away flag
- Opponent defensive strength
- Odds-derived clean sheet and goal probability
- Minutes played ratio (availability indicator)

**Target:** Actual FPL points scored in the gameweek

The optimizer then takes the predicted xP values and solves for the maximum-scoring squad subject to FPL's constraints (budget, position limits, max 3 players per club).

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Model | XGBoost Regressor |
| Training seasons | 2021/22, 2022/23, 2023/24 |
| Validation | Hold-out on 2024/25 GWs |
| MAE | *Add yours here* |
| RMSE | *Add yours here* |

---

## 🗺️ Roadmap

- [ ] Captain and vice-captain optimization
- [ ] Chip strategy (Triple Captain, Free Hit, Bench Boost) planner
- [ ] Mini-league rank tracker
- [ ] M-Pesa paywall integration for premium access
- [ ] Mobile-responsive UI

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 👤 Author

**Abdi Hussein Bachu**
BSc Data Science — KCA University, Nairobi

[![GitHub](https://img.shields.io/badge/GitHub-Bachu33-181717?style=flat&logo=github)](https://github.com/Bachu33)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ⚽ data and 🧠 machine learning. Not affiliated with the official Fantasy Premier League.*
