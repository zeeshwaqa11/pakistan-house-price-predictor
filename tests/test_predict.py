import numpy as np
import pandas as pd
import pytest

from house_prices import config, predict


@pytest.fixture()
def bundle(pipeline_env):
    return predict.load_bundle()


def first_location(bundle, city=None):
    city = city or bundle.cities[0]
    return city, bundle.locations_for(city)[0]


def test_missing_model_gives_friendly_error(tmp_root):
    assert not predict.model_files_exist()
    with pytest.raises(predict.ModelNotFoundError) as info:
        predict.load_bundle()
    assert "python -m house_prices.train" in str(info.value)


def test_prediction_is_positive_and_ordered(bundle):
    for city in bundle.cities:
        location = bundle.locations_for(city)[0]
        for ptype in bundle.property_types[:3]:
            for unit, size in (("Marla", 5), ("Kanal", 1), ("Sq. Ft.", 1200)):
                p = predict.predict(bundle, city, location, ptype, size, unit, 3, 2)
                assert p.estimate > 0
                assert p.lower > 0
                assert p.lower <= p.estimate <= p.upper
                assert p.ppsf == pytest.approx(p.estimate / p.area_sqft)


def test_units_are_converted_with_the_models_marla_factor(bundle):
    city, location = first_location(bundle)
    marla = predict.predict(bundle, city, location, "House", 5, "Marla", 3, 2)
    assert marla.area_sqft == pytest.approx(5 * bundle.marla_sqft)
    kanal = predict.predict(bundle, city, location, "House", 1, "Kanal", 3, 2)
    assert kanal.area_sqft == pytest.approx(20 * bundle.marla_sqft)
    same = predict.predict(bundle, city, location, "House", 5 * bundle.marla_sqft, "Sq. Ft.", 3, 2)
    assert same.estimate == pytest.approx(marla.estimate)


def test_inputs_frame_matches_training_columns(bundle):
    city, location = first_location(bundle)
    X = predict.build_input(bundle, city, location, "House", 5, "Marla", 3, 2)
    assert list(X.columns) == bundle.metadata["features"]["input_columns"]
    assert X["latitude"].notna().all()


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"bedrooms": 0}, "bedrooms"),
        ({"bedrooms": 40}, "bedrooms"),
        ({"baths": 0}, "baths"),
        ({"baths": 2.5}, "baths"),
        ({"area_value": 0.001}, "Size must be"),
        ({"area_value": 1e9}, "Size must be"),
        ({"area_unit": "Acre"}, "Unknown area unit"),
        ({"city": "Atlantis"}, "not one the model"),
        ({"property_type": "Castle"}, "not one the model"),
        ({"location": "Nowhere Town"}, "not found"),
    ],
)
def test_validation_rejects_bad_inputs(bundle, kwargs, fragment):
    city, location = first_location(bundle)
    args = {
        "city": city,
        "location": location,
        "property_type": "House",
        "area_value": 5,
        "area_unit": "Marla",
        "bedrooms": 3,
        "baths": 2,
    }
    args.update(kwargs)
    problems = predict.validate_inputs(bundle, **args)
    assert problems and any(fragment in p for p in problems)
    with pytest.raises(ValueError):
        predict.predict(bundle, **args)


def test_rare_location_is_flagged(bundle):
    index = bundle.location_index
    rare = index[index["n_train"] < config.RARE_LOCATION_COUNT]
    common = index[index["n_train"] >= config.RARE_LOCATION_COUNT]
    if not rare.empty:
        row = rare.iloc[0]
        p = predict.predict(bundle, row["city"], row["location"], "House", 5, "Marla", 3, 2)
        assert p.is_rare
    if not common.empty:
        row = common.iloc[0]
        p = predict.predict(bundle, row["city"], row["location"], "House", 5, "Marla", 3, 2)
        assert not p.is_rare


def test_comparables_are_close_and_limited(bundle):
    listings = pd.read_parquet(config.PATHS.listings)
    city, location = first_location(bundle)
    p = predict.predict(bundle, city, location, "House", 5, "Marla", 3, 2)
    comps = predict.find_comparables(listings, p)
    assert 0 < len(comps) <= config.COMPARABLES
    available = int((listings["location"] == location).sum())
    assert int(comps["same_location"].sum()) >= min(config.COMPARABLES, available) - 1
    assert comps["price"].gt(0).all()
    gap = np.abs(np.log(comps["area_sqft"] / p.area_sqft))
    assert gap.max() < 2.0


def test_comparables_fall_back_to_nearby_locations(bundle):
    listings = pd.read_parquet(config.PATHS.listings)
    city, location = first_location(bundle)
    in_city = listings[listings["city"] == city]
    keep = pd.concat([in_city[in_city["location"] == location].head(2), in_city[in_city["location"] != location]])
    p = predict.predict(bundle, city, location, "House", 5, "Marla", 3, 2)
    comps = predict.find_comparables(keep, p)
    assert len(comps) == config.COMPARABLES
    assert (~comps["same_location"]).any()


def test_presets_are_valid_and_predictable(bundle):
    presets = predict.preset_examples(bundle)
    assert 2 <= len(presets) <= 3
    for preset in presets:
        p = predict.predict(
            bundle,
            preset["city"],
            preset["location"],
            preset["property_type"],
            preset["size"],
            preset["unit"],
            preset["bedrooms"],
            preset["baths"],
        )
        assert p.estimate > 0


def test_explanation_for_a_prediction(bundle):
    city, location = first_location(bundle)
    p = predict.predict(bundle, city, location, "House", 5, "Marla", 3, 2)
    result = predict.explain_inputs(bundle, p, "5 Marla (1,361 sq ft)")
    assert result["sentence"]
    assert result["figure"] is not None
    assert result["prediction_log"] == pytest.approx(float(bundle.pipeline.predict(p.inputs)[0]), abs=1e-3)
