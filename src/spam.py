"""SMS spam classification: data loading, model definitions, training helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

DATA = Path(__file__).resolve().parents[1] / "data" / "sms.tsv"


def load_data(path: Path = DATA) -> pd.DataFrame:
    """Load the SMS Spam Collection and drop exact duplicate messages.

    The raw file has 5,574 messages but 403 are exact repeats. Keeping them
    lets the same message land in both the training and the test set, which
    inflates scores, so they are removed before any split.
    """
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=3)
    df = df.drop_duplicates("text").reset_index(drop=True)
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def normalise(text: str) -> str:
    """Lower-case and replace things whose exact value does not matter."""
    t = text.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " urltoken ", t)
    t = re.sub(r"\b\d{5,}\b", " longnumtoken ", t)      # phone / short codes
    t = re.sub(r"[£$€]\s?\d+", " moneytoken ", t)
    t = re.sub(r"\d+", " numtoken ", t)
    return t


def features() -> FeatureUnion:
    """Word 1-2 grams plus character 3-5 grams (robust to misspellings like 'fr33')."""
    return FeatureUnion([
        ("words", TfidfVectorizer(preprocessor=normalise, ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True)),
        ("chars", TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=2, sublinear_tf=True)),
    ])


def models() -> dict:
    return {
        "Naive Bayes": make_pipeline(
            TfidfVectorizer(preprocessor=normalise, ngram_range=(1, 2), min_df=2), MultinomialNB(alpha=0.1)),
        "Logistic Regression": make_pipeline(
            features(), LogisticRegression(C=20, class_weight="balanced", max_iter=3000)),
        "Linear SVM": make_pipeline(
            features(), CalibratedClassifierCV(LinearSVC(C=1, class_weight="balanced"), cv=3)),
    }
