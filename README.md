# World Happiness Report — Analysis & Predictions (2011–2030)

End-to-end analysis of the World Happiness Report dataset covering 168 countries across 14 years, with machine learning and neural network forecasts to 2030.

---

## Project structure

```
├── world_happiness_report_2005_2025.csv    # Raw data (source)
├── happiness_clean.csv                     # Cleaned dataset (generated)
├── happiness_forecast_2026.csv             # ML forecast for 2026 (generated)
├── happiness_neural_forecast_2026_2030.csv # Neural forecast 2026–2030 (generated)
│
├── main_analysis.ipynb                     # Step 1 — data cleaning & feature engineering
├── visualisation.ipynb                     # Step 2 — charts and exploration
├── predictions.ipynb                       # Step 3 — ML predictions (Linear, Ridge, RF)
├── neural_language_predictions.ipynb       # Step 4 — LSTM & Transformer forecasts
└── Russia_analysis_forecast.ipynb          # Deep-dive: Russia analysis & forecast
```

---

## Dataset

**Source:** World Happiness Report (Gallup World Poll)  
**Kaggle:** [World Happiness Report 2005–2025](https://www.kaggle.com/datasets/elvisbui/world-happiness-report-2005-2025-panel?resource=download)  
**Coverage:** 2011–2025, 168 countries, 2,116 rows

| Column | Description |
|---|---|
| `year` | Survey year |
| `rank_in_year` | Country rank within that year |
| `country` | Country name |
| `happiness_score` | Cantril ladder score (0–10) |
| `lower_whisker` / `upper_whisker` | 95% confidence interval (2019+ only) |
| `explained_log_gdp_per_capita` | GDP contribution to score |
| `explained_social_support` | Social support contribution |
| `explained_healthy_life_expectancy` | Health contribution |
| `explained_freedom` | Freedom to make life choices |
| `explained_generosity` | Generosity contribution |
| `explained_corruption` | Absence of corruption contribution |
| `dystopia_plus_residual` | Unexplained remainder |

> Factor breakdown columns are only available from **2019 onward**. Earlier years have the raw happiness score only.

---

## Notebooks

### `main_analysis.ipynb` — run this first
Cleans the raw CSV and produces `happiness_clean.csv` used by all other notebooks.

- Type casting and missing-value audit
- Country name standardisation (e.g. `Republic of Korea` → `South Korea`)
- Duplicate and sanity checks (score range, rank uniqueness, component-sum verification)
- Adds `tier`, `has_breakdown`, and `score_change_yoy` columns

---

### `visualisation.ipynb`
Six charts covering the full dataset.

---

### `predictions.ipynb`
Traditional ML models benchmarked with time-series cross-validation.

| Model | MAE | R² |
|---|---|---|
| Linear Regression | 0.1085 | 0.981 |
| Ridge Regression | 0.1085 | 0.981 |
| Random Forest | 0.1516 | 0.961 |
| Hist. Gradient Boosting | 0.1689 | 0.951 |

Linear/Ridge win because happiness scores are highly autocorrelated — last year's score is the dominant predictor. Outputs `happiness_forecast_2026.csv`.

---

### `neural_language_predictions.ipynb`
Two neural architectures trained on 4-year rolling windows.

| Model | MAE | R² |
|---|---|---|
| LSTM (2-layer, hidden=64) | 0.265 | 0.909 |
| **Transformer** (d=32, 4 heads) | **0.140** | **0.973** |

The Transformer outperforms LSTM by ~2×, handling the dataset's uneven year gaps better via attention. Produces autoregressive forecasts to 2030 for all 147 countries with 2025 data, saved to `happiness_neural_forecast_2026_2030.csv`.

**2030 predicted top 5:**

| Rank | Country | Score |
|---|---|---|
| 1 | Finland | 7.45 |
| 2 | Iceland | 7.41 |
| 3 | Denmark | 7.39 |
| 4 | Costa Rica | 7.34 |
| 5 | Sweden | 7.32 |

---

### `Russia_analysis_forecast.ipynb`
Country deep-dive with 10 sections.

**Russia forecast (ensemble):**

| Year | Score |
|---|---|
| 2026 | 5.564 |
| 2027 | 5.425 |
| 2028 | 5.336 |
| 2029 | 5.297 |
| 2030 | 5.306 |

Both models agree Russia trends downward from 2026, likely settling near its 2018–2021 range.

---

## Setup

### Install

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Run order

```
1. main_analysis.ipynb                 # generates happiness_clean.csv
2. visualisation.ipynb                 # requires happiness_clean.csv
3. predictions.ipynb                   # requires happiness_clean.csv
4. neural_language_predictions.ipynb   # requires happiness_clean.csv
5. Russia_analysis_forecast.ipynb      # requires happiness_clean.csv
```

Notebooks 2–5 are independent of each other once step 1 is complete.

---

## Key findings

- **Nordic dominance is stable** — Finland, Denmark, Iceland, Sweden have held the top spots every year since 2013 with no signs of displacement by 2030.
- **GDP alone doesn't explain happiness** — the `dystopia_plus_residual` term often exceeds the GDP contribution for top-ranked countries, pointing to cultural and social factors the model cannot fully capture.
- **The 2022 Russia paradox** — social support and perceived freedom *rose* in Russian survey data immediately after the Ukraine invasion, consistent with documented wartime rally effects in self-reported wellbeing surveys.
- **Kazakhstan has overtaken Russia** every year in the dataset and the gap is widening (+0.8 points in 2025). Poland shows the same pattern.
- **Corruption and generosity are Russia's biggest drags** — both sit far below peer countries and show minimal improvement over 14 years.
- **Transformer > LSTM** for this dataset — attention handles missing years and long-range dependencies better than sequential memory, cutting MAE by ~47%.
