# Pakistan House Price Predictor

Estimate the **asking price** of a property in Pakistan from its size, location, type, bedrooms and bathrooms, with an
80% price range, an explanation of what drove the estimate, and five comparable listings, in a local Streamlit app.

The project is about doing the data science carefully: documented cleaning rules with a funnel table, leakage-safe
features, two evaluation setups (random and time-based), baselines every model must beat, error analysis, prediction
intervals with measured coverage, and SHAP explanations.

> **Status of the numbers in this repository.** The real dataset needs a Kaggle login, so it is not included and
> the results tables below hold placeholders until you run the pipeline on the real file. Every table is generated from
> `reports/metrics.json` by code; nothing is typed in by hand. Runs on the synthetic sample (used only for tests and
> CI) are labelled as synthetic and must never be reported.

## Table of contents

- [Data](#data)
- [Pipeline](#pipeline)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Results](#results)
- [Cleaning funnel](#cleaning-funnel)
- [Key figures](#key-figures)
- [Random split vs time split](#random-split-vs-time-split)
- [Design decisions](#design-decisions)
- [The app](#the-app)
- [Tests and CI](#tests-and-ci)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Screenshots](#screenshots)

## Data

- **Source:** a public Kaggle dataset of Pakistani property listings scraped from Zameen.com, for example
  "Pakistan House Price Dataset" (roughly 150k+ rows with columns such as `property_type`, `price`, `location`,
  `city`, `province_name`, `latitude`, `longitude`, `baths`, `bedrooms`, `area`, `Area Type`, `Area Size`, `purpose`,
  `date_added`).
- **Licence:** this repository does not redistribute the data. The licence is stated on the Kaggle dataset page you
  download from (I could not verify it offline, so check it there), and the listing content originates from Zameen.com,
  whose terms of use apply to the underlying material. The code in this repository is MIT licensed (see `LICENSE`).
- **No scraper.** The project never scrapes anything. `data/raw`, `data/interim` and `data/processed` are git-ignored.

### Download steps

1. Sign in to Kaggle (free) and download the CSV of the dataset above.
2. Save it as `data/raw/pakistan_property.csv`.
3. If a step is run without the file, it stops with a message saying where to put it.

Column names are not assumed. `python -m house_prices.load` prints the file's real schema, row count and a sample, and
the mapping from your file's columns to the project's canonical names. If a column is not found, edit
`COLUMN_ALIASES` in `src/house_prices/config.py`; lookups ignore case, spaces and underscores.

### Synthetic sample (tests and CI only)

`python -m scripts.make_sample_data` writes `data/sample/synthetic_pakistan_property.csv`: fake listings with the
same schema and deliberately messy rows. Every row has `agency = "SYNTHETIC DATA"`, which the pipeline detects, and
any report, model metadata or app page built from it says so.

## Pipeline

```mermaid
flowchart LR
    A[data/raw/pakistan_property.csv] --> B[load.py<br/>schema audit + column mapping]
    B --> C[clean.py + units.py<br/>For Sale only, Marla/Kanal to sq ft,<br/>Lakh/Crore text prices, rules, dedupe]
    C --> D[(data/processed/listings.parquet)]
    C --> R1[reports/data_quality.md<br/>cleaning funnel]
    D --> E[split.py<br/>random 80/20 by city<br/>time: newest 20% is test]
    E --> F[train.py<br/>baselines + 5 models<br/>5-fold CV tuning on train only]
    F --> G[(models/ pipeline + metadata)]
    E --> H[evaluate.py<br/>test set touched here first]
    F --> H
    H --> R2[reports/metrics.json<br/>+ figures]
    F --> I[intervals.py<br/>LightGBM quantile 10 / 90]
    G --> J[explain.py<br/>SHAP TreeExplainer]
    I --> R2
    J --> R2
    G --> K[app/streamlit_app.py]
    D --> K
    R2 --> K
```

## Setup

Python 3.11 is the target (the pins in `requirements.txt` support 3.11 and later). In PowerShell, from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that
window and activate again. In VS Code, pick `.venv` as the interpreter (Ctrl+Shift+P, "Python: Select Interpreter").
Everything runs locally with free tools; there are no API keys.

## How to run

Every step is a `python -m` command and each one checks that its inputs exist.

```powershell
python -m house_prices.load          # print the real schema, row count, sample and column mapping
python -m house_prices.clean         # rules, funnel, data/processed/listings.parquet, reports/data_quality.md
python -m house_prices.split         # random and time splits (ids only, data untouched)
python -m house_prices.train         # baselines, five tuned models, model files and metadata
python -m house_prices.evaluate      # first use of the test set: metrics, error analysis, figures
python -m house_prices.intervals     # 80% prediction intervals and their coverage
python -m house_prices.explain       # SHAP figures and summary
python -m scripts.update_readme      # fill the tables in this README and the model card from metrics.json
python -m streamlit run app/streamlit_app.py
```

Or run everything in order (add `--notebooks` to build and execute the four notebooks in place):

```powershell
python -m scripts.run_all
python -m scripts.run_all --notebooks
```

Notebooks are generated by `python -m scripts.build_notebooks` and must be executed after the pipeline so their saved
outputs come from the real data:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit.ipynb notebooks/02_eda.ipynb notebooks/03_modelling.ipynb notebooks/04_error_analysis.ipynb
```

Tuning is sized to be laptop-friendly: each search runs on up to 40,000 training rows (`SEARCH_MAX_ROWS` in `train.py`, or `--search-rows N`, `0` for all rows), then the winner is refit on the full training set. Training both splits on 88,000 synthetic training rows took about eight minutes on the development machine. Use `python -m house_prices.train --setup random --models lightgbm` to train a single model.

## Results

Generated from `reports/metrics.json`. Errors are in rupees on the held-out test set; R² is on log price. The
cross-validation column is the mean ± standard deviation of the fold RMSE on log price (lower is better).

<!-- BEGIN:RESULTS -->
_Not generated yet. Run the pipeline on the real dataset (see [How to run](#how-to-run)) and then `python -m scripts.update_readme`._
<!-- END:RESULTS -->

### Prediction interval coverage

An 80% interval should contain the true asking price about 80% of the time. Coverage is measured on the test set, overall
and per city, along with how wide the ranges are.

<!-- BEGIN:INTERVALS -->
_Not generated yet. Run the pipeline on the real dataset (see [How to run](#how-to-run)) and then `python -m scripts.update_readme`._
<!-- END:INTERVALS -->

### Where the model is weakest

<!-- BEGIN:ERRORS -->
_Not generated yet. Run the pipeline on the real dataset (see [How to run](#how-to-run)) and then `python -m scripts.update_readme`._
<!-- END:ERRORS -->

The full breakdown (city, property type, price decile, listings per location), the ten worst predictions with a guessed
reason for each, and the plots are in `notebooks/04_error_analysis.ipynb` and the app's Model performance page.

### Location encoding

<!-- BEGIN:ABLATION -->
_Not generated yet. Run the pipeline on the real dataset (see [How to run](#how-to-run)) and then `python -m scripts.update_readme`._
<!-- END:ABLATION -->

### What drives the estimate

<!-- BEGIN:SHAP -->
_Not generated yet. Run the pipeline on the real dataset (see [How to run](#how-to-run)) and then `python -m scripts.update_readme`._
<!-- END:SHAP -->

## Cleaning funnel

Copied from `reports/data_quality.md`, which `python -m house_prices.clean` writes. Each rule is a small function with its
own unit tests.

<!-- BEGIN:FUNNEL -->
_Not generated yet. Run the pipeline on the real dataset (see [How to run](#how-to-run)) and then `python -m scripts.update_readme`._
<!-- END:FUNNEL -->

## Key figures

Written to `reports/figures/` by the notebooks and the pipeline.

| Figure | File |
|---|---|
| Median price per sq ft by location (map) | `reports/figures/price_map.png` and interactive `price_map.html` |
| Model comparison, both splits | `reports/figures/model_comparison.png` |
| Predicted vs actual | `reports/figures/predicted_vs_actual_random.png`, `..._time.png` |
| Error by group | `reports/figures/error_by_group_random.png`, `..._time.png` |
| SHAP mean absolute value | `reports/figures/shap_bar.png` |
| SHAP beeswarm | `reports/figures/shap_beeswarm.png` |
| SHAP waterfall for one listing | `reports/figures/shap_waterfall_example.png` |

Once the pipeline has run on the real data, embed the images you like, for example
`![Model comparison](reports/figures/model_comparison.png)`.

## Random split vs time split

Both are reported side by side.

1. **Random split.** Listings are shuffled and 80% train, 20% test, stratified by city, with a fixed `random_state`.
   The test set contains listings from the same weeks, agents and near-duplicate advertisements as the training set, so the
   model is partly being asked to recognise what it has already seen. This tends to give the more flattering score.
2. **Time split.** Listings are ordered by `date_added`; the model trains on the older listings and is tested on the
   newest 20%. This mimics real use, where the model prices listings that appear after it was trained. It usually gives a
   worse, more honest score, especially if the market moved over the period.

The test set is created first and is read for the first time in `evaluate.py`. `train.py` loads training rows only
(`split.load_train`), and a test checks that changing test-row prices does not change a trained model. Cross-validation
inside the training set is random 5-fold in both setups, so for the time split it is more optimistic than the time-split
test; that gap is exactly what is being shown.

## Design decisions

- **Only "For Sale" listings.** Rentals are a different problem with different price scales.
- **Areas in sq ft.** Marla, Kanal, sq yd, sq ft and sq m are converted in `units.py`. A Marla has no single official
  size, so the factor is a config value (`MARLA_SQFT_SETTING`). The default `"auto"` tries to infer it from the data
  (locations that list both Marla and sq ft/sq yd areas) and accepts it only inside 200-300 sq ft; otherwise it falls back to
  272.25 sq ft. Kanal is 20 Marla. The factor used, and where it came from, is written to `reports/data_quality.md` and
  `models/model_metadata.json`, and the app uses the same factor. If the raw file only uses Marla and Kanal (no other unit to
  compare against), the fallback applies, and changing the setting rescales all sizes consistently.
- **Prices.** Text prices such as "1.5 Crore" or "85 Lakh" are converted to rupees (`units.parse_price`); the count is in the report.
- **Robust outlier rules.** Price per sq ft is checked against a per-city IQR fence on log price per sq ft, not a global
  cutoff. Because that fence uses price, it is computed on the whole cleaned file before splitting: the evaluation describes
  plausible listings, not every raw record.
- **Target is log(price).** Predictions are converted back to rupees for reporting.
- **No target leakage.** Price per sq ft and anything price-derived is never a feature. `FeatureBuilder` reads only the eight
  input columns, and tests fail if a feature is derived from the target.
- **Location handling.** Locations with fewer than `MIN_LOCATION_COUNT` training listings become "Other (city)"; the
  rest are target-encoded with smoothing and 5-fold cross-fitting, inside the scikit-learn `Pipeline`, so the encoder is refit
  on each training fold only. A one-hot alternative is compared in the results.
- **Deployed model.** The best tree model (Random Forest, XGBoost or LightGBM) by cross-validated log RMSE, trained on the
  random-split training set so the reported test scores describe exactly the deployed model. Linear models are benchmarks;
  the app's SHAP explanations need a tree model.
- **Intervals.** Two LightGBM quantile models (10th and 90th percentiles) use the tuned LightGBM parameters. The range is
  clipped so that it always contains the point estimate. Coverage is reported honestly, whether or not it reaches 80%.
- **Explanations.** SHAP `TreeExplainer`, with one-hot and encoded columns mapped to readable names and grouped into
  Location, Size, City, Property type, Bedrooms, Bathrooms, Coordinates and Distance to centre for the plain-English sentence.

## The app

```powershell
python -m streamlit run app/streamlit_app.py
```

- **Predict:** city, searchable location list filtered to the city, property type, size with Marla / Kanal / sq ft,
  bedrooms and bathrooms, three example presets. Shows the estimate and 80% range in Lakh/Crore with exact rupees beneath,
  price per sq ft, a SHAP waterfall with a plain-English sentence, five comparable listings, and a warning for rare locations.
- **Market explorer:** price map and price-per-sq-ft charts with a city filter.
- **Model performance:** tables and charts read from `reports/metrics.json`.
- **About & limitations.**

The model is loaded once with `st.cache_resource`. If the model files are missing, the page shows which commands to run.
Every page carries the disclaimer that estimates are based on historical asking prices, are for learning, and are not
financial or property advice.

## Tests and CI

```powershell
python -m pytest
python -m ruff check .
```

The suite runs on synthetic data only: unit conversion (including a configurable Marla factor), Lakh/Crore parsing and
formatting, each cleaning rule on small handmade frames, leakage checks (target encoding inside the pipeline, test set
untouched, no price-derived feature), the time split, prediction ordering (`lower <= estimate <= upper`), and Streamlit
`AppTest` smoke tests for every page. `.github/workflows/ci.yml` runs ruff and pytest on Python 3.11.

## Project structure

```text
pakistan-house-prices/
  data/raw, data/interim, data/processed   (git-ignored; raw file goes in data/raw)
  notebooks/     01_data_audit, 02_eda, 03_modelling, 04_error_analysis
  src/house_prices/
    config.py    paths, column aliases, thresholds, Marla setting, city centres
    load.py      read the CSV, print the schema, map columns
    units.py     Marla/Kanal/sq yd conversion, Lakh/Crore price parsing, haversine
    clean.py     cleaning rules, funnel, data_quality.md
    features.py  feature builder, location grouping, preprocessing pipeline
    split.py     random and time splits
    train.py     baselines, tuning, model files, metadata
    evaluate.py  metrics, error analysis, figures, metrics.json
    intervals.py 80% prediction intervals and coverage
    explain.py   SHAP global and local explanations
    predict.py   load the model, validate inputs, predict, comparables
    formatting.py PKR, Lakh and Crore formatting
  app/           streamlit_app.py and pages/
  models/        house_price_model.joblib, interval models, model_metadata.json, location_index.csv
  reports/       figures/, metrics.json, data_quality.md, model_card.md
  scripts/       make_sample_data, build_notebooks, run_all, update_readme
  tests/
```

## Limitations

- **Asking prices, not sale prices.** Listings usually ask for more than the final deal, by an amount that varies.
- **A limited time period.** The data covers a short window of listings. Prices have since moved with inflation, so today's
  market can sit well above anything the model has seen; the time split shows how much performance degrades even within the window.
- **Uneven coverage across cities.** Large cities and popular locations have many listings, while small cities and rare
  locations have few. Errors are larger there, and the app warns when a location is rare.
- **Marla ambiguity.** Local Marla sizes differ (225 and 272.25 sq ft are both used), so every Marla-denominated size, and
  therefore price per sq ft, is only as accurate as the factor chosen.
- **Missing determinants of price.** Age, condition, floor, furnishing, corner plots and road width are not in the data.
- **Cleaning shapes the population.** Outlier and duplicate rules use price and are applied before splitting. Near-duplicate
  removal can also remove genuinely distinct units.
- **Duplicated listings across splits.** Reposted advertisements that differ slightly can appear on both sides of a random split.
- **Not advice.** The project is for learning.

## Future work

- Evaluate on a later scrape to measure drift directly, and inflation-adjust prices.
- Try conformalised quantile regression to tighten interval coverage in small cities.
- Add listing text features and image-based signals, if a source with permission is available.
- Rolling-origin (expanding window) evaluation instead of a single time split.
- Per-city models or a hierarchical location effect for thin locations.
- Monitoring of input drift in the app.

## Screenshots

Capture these once the model has been trained on the real data (save into `reports/figures/screenshots/`):

1. **Predict** with an example preset selected: estimate, range, waterfall and comparables visible.
2. **Predict** with a rare location selected, showing the reliability warning.
3. **Market explorer** with the map and a city filter applied.
4. **Model performance** showing the random and time split tables.
5. **About & limitations.**

Screenshots: _to be added_.

## Licence

MIT for the code (see `LICENSE`). See [Data](#data) for the dataset.
