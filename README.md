# Eid al-Adha Livestock Price Forecast

End-to-end data science project to forecast weekly sheep & goat meat prices using EU Agrifood API data, framed around Eid al-Adha seasonality.

## Project structure

```text
eid-livestock-forecast/
├── README.md
├── requirements.txt
├── data/
│   └── sheep_goat_prices_eu.csv
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── features.py
│   ├── modeling.py
│   └── evaluation.py
├── models/
│   └── (saved models, pipelines)
└── app/
    └── app.py
```

## Quickstart

```bash
# 1. Create and activate a virtualenv (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a full experiment (prep + train + evaluate)
python -m src.modeling

# 4. Launch the dashboard
streamlit run app/app.py
```

The dataset in `data/sheep_goat_prices_eu.csv` contains EU weekly prices for heavy and light lamb by member state and week. We engineer calendar features and an `eid_season` flag (you can extend the Eid calendar as you like).
