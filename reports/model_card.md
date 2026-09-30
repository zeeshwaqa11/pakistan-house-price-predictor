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
_Not generated yet. Run the pipeline on the real dataset and then `python -m scripts.update_readme`._
<!-- END:RESULTS -->

### 80% prediction interval

<!-- BEGIN:INTERVALS -->
_Not generated yet. Run the pipeline on the real dataset and then `python -m scripts.update_readme`._
<!-- END:INTERVALS -->

### Weakest areas

<!-- BEGIN:ERRORS -->
_Not generated yet. Run the pipeline on the real dataset and then `python -m scripts.update_readme`._
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
