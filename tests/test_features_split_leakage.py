import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from house_prices import clean, config, features, load, split


@pytest.fixture(scope="module")
def listings(synthetic_raw):
    cleaned, *_ = clean.clean(load.apply_schema(synthetic_raw))
    return cleaned


def test_feature_builder_columns_and_values(listings):
    out = features.FeatureBuilder().fit_transform(features.make_inputs(listings))
    for column in features.NUMERIC_FEATURES + features.CATEGORICAL_FEATURES + ["location"]:
        assert column in out.columns
    assert np.allclose(out["log_area_sqft"], np.log(listings["area_sqft"]))
    assert (out["distance_to_centre_km"].dropna() >= 0).all()
    assert out["distance_to_centre_km"].dropna().max() < 150


def test_distance_to_centre_is_zero_at_the_centre():
    frame = pd.DataFrame(
        {
            "area_sqft": [1000.0],
            "bedrooms": [3],
            "baths": [2],
            "property_type": ["House"],
            "city": ["Lahore"],
            "location": ["X"],
            "latitude": [config.CITY_CENTRES["Lahore"][0]],
            "longitude": [config.CITY_CENTRES["Lahore"][1]],
        }
    )
    out = features.FeatureBuilder().transform(frame)
    assert out["distance_to_centre_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_unknown_city_gives_missing_distance():
    frame = pd.DataFrame(
        {
            "area_sqft": [1000.0],
            "bedrooms": [3],
            "baths": [2],
            "property_type": ["House"],
            "city": ["Atlantis"],
            "location": ["X"],
            "latitude": [30.0],
            "longitude": [70.0],
        }
    )
    assert np.isnan(features.FeatureBuilder().transform(frame)["distance_to_centre_km"].iloc[0])


def test_no_feature_is_computed_from_the_target(listings):
    inputs = features.make_inputs(listings)
    for name in features.TARGET_DERIVED_NAMES:
        assert name not in inputs.columns
        assert name not in features.INPUT_COLUMNS
        assert name not in features.NUMERIC_FEATURES + features.CATEGORICAL_FEATURES
    builder = features.FeatureBuilder()
    with_price = builder.transform(listings)
    scrambled = listings.assign(price=np.random.default_rng(0).uniform(1e5, 1e9, len(listings)))
    without_price = builder.transform(scrambled.drop(columns=["price"]))
    pd.testing.assert_frame_equal(with_price, without_price)
    assert features.make_target(scrambled).equals(np.log(scrambled["price"]))


def test_pipeline_cannot_see_price_column(listings):
    pipeline = features.build_pipeline(Ridge())
    y = features.make_target(listings)
    pipeline.fit(features.make_inputs(listings), y)
    out = pipeline[:-1].transform(features.make_inputs(listings))
    names = pipeline.named_steps["prep"].get_feature_names_out()
    assert not any(name in n for n in names for name in ("price", "ppsf"))
    assert out.shape[0] == len(listings)


def test_location_grouper_groups_rare_locations_per_city():
    frame = pd.DataFrame(
        {
            "city": ["Lahore"] * 5 + ["Lahore"] * 2 + ["Karachi"] * 2,
            "location": ["A"] * 5 + ["B"] * 2 + ["C"] * 2,
        }
    )
    grouper = features.LocationGrouper(min_count=3).fit(frame)
    out = grouper.transform(frame)
    assert out[features.LOCATION_FEATURE].tolist() == (
        ["Lahore | A"] * 5 + ["Other (Lahore)"] * 2 + ["Other (Karachi)"] * 2
    )


def test_location_grouper_is_fitted_on_training_data_only():
    train = pd.DataFrame({"city": ["Lahore"] * 5, "location": ["A"] * 5})
    test = pd.DataFrame({"city": ["Lahore"] * 5, "location": ["OnlyInTest"] * 5})
    grouper = features.LocationGrouper(min_count=3).fit(train)
    assert "Lahore | OnlyInTest" not in grouper.frequent_
    assert set(grouper.transform(test)[features.LOCATION_FEATURE]) == {"Other (Lahore)"}


def test_target_encoding_is_inside_the_pipeline_and_fit_on_train_only(listings):
    pipeline = features.build_pipeline(Ridge())
    assert isinstance(pipeline, Pipeline)
    assert any(isinstance(t, TargetEncoder) for _, t, _ in pipeline.named_steps["prep"].transformers)
    train, test = split.random_split(listings)
    train_df, test_df = listings.loc[train], listings.loc[test]
    only_test = test_df.copy()
    only_test["location"] = "Location That Exists Only In Test"
    pipeline.fit(features.make_inputs(train_df), features.make_target(train_df))
    fitted = pipeline.named_steps["prep"].named_transformers_["loc"]
    learned = set(fitted.categories_[0])
    train_keys = set(
        pipeline.named_steps["location"].transform(
            pipeline.named_steps["features"].transform(features.make_inputs(train_df))
        )[features.LOCATION_FEATURE]
    )
    assert learned <= train_keys
    transformed = pipeline.named_steps["location"].transform(
        pipeline.named_steps["features"].transform(features.make_inputs(only_test))
    )
    assert transformed[features.LOCATION_FEATURE].str.startswith("Other (").all()


def test_target_encoding_does_not_depend_on_test_targets(listings):
    train_idx, test_idx = split.random_split(listings)
    train_df = listings.loc[train_idx]
    test_df = listings.loc[test_idx]
    X_train, y_train = features.make_inputs(train_df), features.make_target(train_df)
    a = features.build_pipeline(Ridge()).fit(X_train, y_train)
    b = features.build_pipeline(Ridge()).fit(X_train, y_train)
    changed = test_df.assign(price=test_df["price"] * 50)
    X_test = features.make_inputs(test_df)
    pa = a.predict(X_test)
    pb = b.predict(features.make_inputs(changed))
    np.testing.assert_allclose(pa, pb)


def test_random_split_is_disjoint_complete_and_stratified(listings):
    train, test = split.random_split(listings)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(listings)
    assert len(test) / len(listings) == pytest.approx(config.TEST_SIZE, abs=0.01)
    overall = listings["city"].value_counts(normalize=True)
    in_test = listings.loc[test, "city"].value_counts(normalize=True)
    for city in overall.index:
        assert in_test[city] == pytest.approx(overall[city], abs=0.02)


def test_random_split_is_reproducible(listings):
    a = split.random_split(listings)
    b = split.random_split(listings)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = split.random_split(listings, random_state=1)
    assert not np.array_equal(a[1], c[1])


def test_time_split_puts_newer_listings_in_test(listings):
    train, test = split.time_split(listings)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(listings)
    assert listings.loc[train, "date_added"].max() < listings.loc[test, "date_added"].min()
    assert len(test) / len(listings) == pytest.approx(config.TEST_SIZE, abs=0.03)


def test_time_split_on_handmade_dates():
    df = pd.DataFrame({"date_added": pd.date_range("2019-01-01", periods=10)})
    train, test = split.time_split(df, test_size=0.2)
    assert test.tolist() == [8, 9]
    assert train.tolist() == list(range(8))


def test_apply_split_round_trip(listings):
    splits = split.make_splits(listings)
    for setup in split.SETUPS:
        train, test = split.apply_split(listings, splits, setup)
        assert len(train) + len(test) == len(listings)
        assert set(train["listing_id"]).isdisjoint(test["listing_id"])
    with pytest.raises(ValueError):
        split.apply_split(listings, splits, "bogus")
