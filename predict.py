"""Classify one or more SMS messages with the trained model.

    python predict.py "WINNER!! Claim your free prize now, call 09061701461"
    python predict.py "Are we still meeting at 6?" "URGENT! Your mobile won £2000"
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import spam  # noqa: E402,F401  (needed to unpickle the custom preprocessor)


def main(messages):
    model = pickle.load(open(Path(__file__).parent / "models" / "spam_model.pkl", "rb"))
    for msg, p in zip(messages, model.predict_proba(messages)[:, 1]):
        label = "SPAM" if p >= 0.5 else "ham "
        print(f"{label}  {p:5.1%}  {msg}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
