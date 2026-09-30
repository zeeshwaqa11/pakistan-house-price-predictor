import pandas as pd
import pytest

from house_prices import config, load


def test_missing_file_gives_download_instructions(tmp_root):
    with pytest.raises(load.DataMissingError) as info:
        load.load_raw()
    message = str(info.value)
    assert "pakistan_property.csv" in message
    assert "Kaggle" in message


def test_main_returns_nonzero_when_file_missing(tmp_root, capsys):
    assert load.main([]) == 1
    assert "Dataset not found" in capsys.readouterr().err


def test_alias_resolution_ignores_case_and_spacing():
    df = pd.DataFrame(columns=["Area Size", "AREA TYPE", "Property_Type", "price"])
    mapping = load.resolve_columns(df)
    assert mapping["area_size"] == "Area Size"
    assert mapping["area_unit"] == "AREA TYPE"
    assert mapping["property_type"] == "Property_Type"


def test_apply_schema_reports_missing_columns():
    df = pd.DataFrame({"price": [1], "city": ["Lahore"]})
    with pytest.raises(load.SchemaError) as info:
        load.apply_schema(df)
    assert "Columns present" in str(info.value)
    assert "COLUMN_ALIASES" in str(info.value)


def test_apply_schema_on_synthetic(synthetic_raw):
    out = load.apply_schema(synthetic_raw)
    assert list(out.columns) == list(config.COLUMN_ALIASES)
    assert len(out) == len(synthetic_raw)


def test_synthetic_detection(synthetic_raw):
    assert load.is_synthetic(load.apply_schema(synthetic_raw))
    real_like = load.apply_schema(synthetic_raw).assign(agency="Some Agency")
    assert not load.is_synthetic(real_like)


def test_schema_report_mentions_row_count(synthetic_raw):
    report = load.schema_report(synthetic_raw)
    assert f"Rows: {len(synthetic_raw):,}" in report


def test_file_hash_is_stable(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("a,b\n1,2\n")
    assert load.file_sha256(path) == load.file_sha256(path)
    assert len(load.file_sha256(path)) == 64
