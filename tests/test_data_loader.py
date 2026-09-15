"""
Unit tests for src/data_loader.py
"""

import os
import pytest
import pandas as pd
from src.data_loader import load_applesupport_data, generate_synthetic_applesupport_threads


def test_generate_synthetic_applesupport_threads():
    df = generate_synthetic_applesupport_threads(sample_size=140, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 140
    assert "thread_id" in df.columns
    assert "customer_tweet" in df.columns
    assert "brand_response" in df.columns
    assert "intent" in df.columns
    assert df["intent"].nunique() >= 7


def test_load_applesupport_data():
    test_csv = "data/test_subsample.csv"
    if os.path.exists(test_csv):
        os.remove(test_csv)

    df = load_applesupport_data(data_path=test_csv, sample_size=50)
    assert len(df) == 50
    assert os.path.exists(test_csv)
    
    # Cleanup
    if os.path.exists(test_csv):
        os.remove(test_csv)
