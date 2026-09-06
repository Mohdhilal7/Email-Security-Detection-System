import pandas as pd
from src.data_utils import infer_columns, normalize_label, clean_for_tfidf

def test_infer_columns():
    df = pd.DataFrame({"Text": ["hello"], "Label": ["ham"]})
    assert infer_columns(df) == ("Text", "Label")

def test_normalize_label():
    assert normalize_label("spam") == 1
    assert normalize_label("ham") == 0
    assert normalize_label(1) == 1
    assert normalize_label(0) == 0

def test_clean_for_tfidf():
    cleaned = clean_for_tfidf("Hello <b>WORLD</b> https://example.com a@b.com")
    assert "urltoken" in cleaned
    assert "emailtoken" in cleaned
    assert "<b>" not in cleaned
