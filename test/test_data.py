import pandas as pd

def test_data_shape_and_types():
    df = pd.read_csv("data.csv")
    assert "feature" in df.columns and "target" in df.columns
    assert df["feature"].dtype.kind in "if" and df["target"].dtype.kind in "if"
    assert len(df) >= 3  # sanity: at least a few rows

