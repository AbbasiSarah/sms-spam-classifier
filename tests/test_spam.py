import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spam import load_data, normalise  # noqa: E402


def test_no_duplicates_and_labels():
    df = load_data()
    assert df.text.duplicated().sum() == 0
    assert set(df.y.unique()) == {0, 1}


def test_normalise_masks_numbers_and_links():
    t = normalise("Call 09061701461 or visit www.win.com for £100")
    assert "longnumtoken" in t and "urltoken" in t and "moneytoken" in t
