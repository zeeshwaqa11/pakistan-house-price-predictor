# Model card: Pakistan House Price Predictor

The tables in this card are filled in from `reports/metrics.json` by `python -m scripts.update_readme`. Until the
pipeline has been run on the real dataset they show placeholders. Nothing here is typed in by hand.

## Intended use

- **Purpose:** estimate the *asking price* of a residential property in Pakistan (PKR) from its size, city, location,
  property type, bedrooms and bathrooms, with an 80% price range and an explanation of what drove the estimate.
- **Intended users:** people learning how a careful tabular-regression project is built and evaluated.
- **Out of scope:** valuations for lending, taxation, insurance, legal disputes or negotiation; predicting sale prices,
  rents or plots; properties outside the cities and property types in the training data; any decision that affects a person's
  finances without an independent valuation.

## Data

- **Source:** a public Kaggle dataset of property listings scraped from Zameen.com (column names are mapped in
  `src/house_prices/config.py`). The data is not redistributed here; see the README for the licence note and download steps.
- **Filtering:** "For Sale" listings only. Areas are converted to sq ft (Marla factor recorded in
  `models/model_metadata.json`), text prices are converted to rupees, duplicates and implausible values are removed by
  documented rules (`reports/data_quality.md`).
- **Period:** the listings cover a limited window (see the date ranges in the results below). Prices in the data are
  historical asking prices.
- **Fields not used:** agent and agency names, page URLs and listing identifiers are not model inputs.

## Model

- **Target:** log(price in PKR); predictions are converted back to rupees.
- **Inputs:** size in sq ft (and its log), bedrooms, bathrooms, property type, city, location (grouped when rare, then
  target-encoded inside the pipeline), latitude, longitude, distance to the city centre.
- **Candidates:** Linear Regression, Ridge, Random Forest, XGBoost, LightGBM, each in a scikit-learn `Pipeline`, tuned
  with 5-fold cross-validation on the training set only. Two baselines (overall median; location median price per sq ft times
  size) are reported next to them.
- **Deployed:** the best tree model by cross-validation, trained on the random-split training set.
- **Range:** LightGBM quantile regression at the 10th and 90th percentiles, clipped to contain the point estimate.

## Metrics

Errors are in rupees on the held-out test set; R² is on log price. Two evaluation setups are reported: a random split
(stratified by city) and a time split (test = newest 20% of listings), which is the more realistic of the two.

<!-- BEGIN:RESULTS -->

**Random split**: train 53,658 listings (2018-08-05 to 2019-08-06), test 13,415 (2018-08-05 to 2019-08-06).

| Model | MAE | RMSE | Median % error | Mean % error | R² (log price) | CV RMSE (log), mean ± std |
|---|---|---|---|---|---|---|
| Baseline: overall median price | PKR 1.76 Crore | PKR 3.96 Crore | 61.4% | 96.8% | -0.006 | 0.996 ± 0.008 |
| Baseline: location median price/sq ft x size | PKR 75.9 Lakh | PKR 3.03 Crore | 21.8% | 35.2% | 0.820 | 0.423 ± 0.005 |
| Linear Regression | PKR 68.16 Lakh | PKR 1.85 Crore | 21.1% | 30.4% | 0.857 | 0.380 ± 0.005 |
| Ridge | PKR 68.44 Lakh | PKR 1.86 Crore | 21.1% | 30.4% | 0.857 | 0.379 ± 0.004 |
| Random Forest | PKR 49.75 Lakh | PKR 1.56 Crore | 14.5% | 21.5% | 0.915 | 0.301 ± 0.004 |
| **XGBoost** | PKR 47.31 Lakh | PKR 1.45 Crore | 13.6% | 19.7% | 0.929 | 0.276 ± 0.004 |
| LightGBM | PKR 47.02 Lakh | PKR 1.51 Crore | 13.7% | 19.8% | 0.928 | 0.277 ± 0.003 |

**Time split**: train 53,524 listings (2018-08-05 to 2019-07-03), test 13,549 (2019-07-04 to 2019-08-06).

| Model | MAE | RMSE | Median % error | Mean % error | R² (log price) | CV RMSE (log), mean ± std |
|---|---|---|---|---|---|---|
| Baseline: overall median price | PKR 1.63 Crore | PKR 3.55 Crore | 62.7% | 118.1% | -0.002 | 0.989 ± 0.004 |
| Baseline: location median price/sq ft x size | PKR 73.47 Lakh | PKR 2.45 Crore | 22.8% | 44.2% | 0.772 | 0.405 ± 0.003 |
| Linear Regression | PKR 64.92 Lakh | PKR 1.57 Crore | 22.0% | 37.5% | 0.821 | 0.367 ± 0.002 |
| Ridge | PKR 64.94 Lakh | PKR 1.57 Crore | 22.0% | 37.5% | 0.821 | 0.367 ± 0.002 |
| Random Forest | PKR 49.1 Lakh | PKR 1.31 Crore | 15.4% | 26.6% | 0.888 | 0.295 ± 0.003 |
| XGBoost | PKR 47.92 Lakh | PKR 1.39 Crore | 14.5% | 23.1% | 0.910 | 0.267 ± 0.004 |
| **LightGBM** | PKR 46.89 Lakh | PKR 1.28 Crore | 14.6% | 23.2% | 0.910 | 0.266 ± 0.004 |

Bold rows are the best tree model by cross-validation, chosen without looking at the test set. Cross-validation used up to 40000 training rows for the tuning search.
<!-- END:RESULTS -->

### 80% prediction interval

<!-- BEGIN:INTERVALS -->

Nominal coverage is 80%.

| Split | Empirical coverage | Mean width | Median width | Mean width / estimate |
|---|---|---|---|---|
| random | 73.8% | PKR 1.29 Crore | PKR 64.59 Lakh | 54% |
| time | 67.8% | PKR 1.12 Crore | PKR 58.71 Lakh | 53% |

Per city, random split:

| City | Test listings | Coverage | Mean width |
|---|---|---|---|
| Faisalabad | 374 | 70.1% | PKR 82.05 Lakh |
| Islamabad | 2,069 | 71.8% | PKR 1.55 Crore |
| Karachi | 5,008 | 75.3% | PKR 1.37 Crore |
| Lahore | 4,624 | 73.2% | PKR 1.27 Crore |
| Rawalpindi | 1,340 | 74.5% | PKR 80.22 Lakh |

Per city, time split:

| City | Test listings | Coverage | Mean width |
|---|---|---|---|
| Faisalabad | 538 | 65.8% | PKR 67.39 Lakh |
| Islamabad | 2,391 | 67.2% | PKR 1.28 Crore |
| Karachi | 5,188 | 67.0% | PKR 1.18 Crore |
| Lahore | 3,741 | 69.0% | PKR 1.21 Crore |
| Rawalpindi | 1,691 | 69.3% | PKR 67.98 Lakh |
<!-- END:INTERVALS -->

### Weakest areas

<!-- BEGIN:ERRORS -->

**Random split**

- By city, the weakest group is 'Faisalabad' with a median error of 16.8% (n=374) against 13.6% overall; the strongest is 'Islamabad' at 12.5%.
- By property type, the weakest group is 'Penthouse' with a median error of 20.7% (n=51) against 13.6% overall; the strongest is 'Lower Portion' at 10.3%.
- By price decile, the weakest group is '1' with a median error of 18.3% (n=1384) against 13.6% overall; the strongest is '7' at 10.4%.
- By location coverage, the weakest group is 'not in training data' with a median error of 21.4% (n=63) against 13.6% overall; the strongest is '200+ listings' at 13.1%.

**Time split**

- By city, the weakest group is 'Karachi' with a median error of 16.8% (n=5188) against 14.6% overall; the strongest is 'Lahore' at 13.0%.
- By property type, the weakest group is 'Penthouse' with a median error of 26.9% (n=45) against 14.6% overall; the strongest is 'House' at 13.9%.
- By price decile, the weakest group is '1' with a median error of 24.8% (n=1357) against 14.6% overall; the strongest is '7' at 12.0%.
- By location coverage, the weakest group is 'not in training data' with a median error of 25.8% (n=76) against 14.6% overall; the strongest is '50-199 listings' at 13.8%.

<!-- END:ERRORS -->

## Limitations

- Asking prices are not sale prices; listings usually ask for more than the final deal.
- The data covers a limited period. Prices have since moved with inflation, so the model will under-estimate a current market.
- Coverage is uneven: large cities and popular locations dominate. Estimates for rare locations and very expensive
  properties are less reliable, and the model tends to pull extreme prices toward the middle.
- A Marla has no single official size; the factor used is stored with the model and changes every Marla-based size.
- Condition, age, floor, furnishing, road width and corner position are not observed.
- Outlier and duplicate rules use price and are applied before splitting, so metrics describe plausible listings.
- Interval coverage is measured, not guaranteed; it can be below 80% for a later period or a thin city.

## Ethical considerations

- **Location is a strong proxy.** Location dominates the estimate, so the model reproduces and can reinforce existing
  price differences between neighbourhoods, which are tied to income and access to services. It should not be used to
  screen, rank or steer people.
- **Scraped content.** The listings come from a commercial site. Check the dataset licence and the site's terms before any use
  beyond learning, and do not use the data to identify sellers or agents.
- **Automation risk.** A confident-looking number can anchor a negotiation. The app therefore shows a range, the
  reasons, and a reliability warning for rare locations, and carries a disclaimer on every page.
- **Personal data.** The model does not use names, contact details or listing text.
- **Synthetic data.** Runs on the synthetic sample used in tests and CI are labelled as such and are never results.
