"""Train and evaluate the spam classifiers; writes results/ and models/.

    python train.py
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))
from spam import load_data, models  # noqa: E402

OUT, FIG = Path("results"), Path("results/figures")
SPAM, HAM, GREY = "#D84315", "#546E7A", "#B0BEC5"


def pretty(w):
    for k, v in {"longnumtoken": "[phone/short code]", "urltoken": "[link]", "moneytoken": "[£ amount]",
                 "numtoken": "[number]"}.items():
        w = w.replace(k, v)
    return w


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"precision": precision_score(y, pred), "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "false_alarms": int(((pred == 1) & (y == 0)).sum()), "missed_spam": int(((pred == 0) & (y == 1)).sum())}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_data()
    train, test = train_test_split(df, test_size=0.2, stratify=df.y, random_state=42)
    print(f"{len(df)} unique messages ({df.y.sum()} spam); train {len(train)}, test {len(test)}")

    results, test_probs = {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, model in models().items():
        p_cv = cross_val_predict(model, train.text, train.y, cv=cv, method="predict_proba")[:, 1]
        model.fit(train.text, train.y)
        p_test = model.predict_proba(test.text)[:, 1]
        results[name] = {"cv": metrics(train.y.values, p_cv), "test": metrics(test.y.values, p_test)}
        test_probs[name] = p_test
        t = results[name]["test"]
        print(f"{name:20s} test F1 {t['f1']:.3f}  precision {t['precision']:.3f}  recall {t['recall']:.3f}  "
              f"false alarms {t['false_alarms']}  missed {t['missed_spam']}")

    best = max(results, key=lambda m: results[m]["cv"]["f1"])  # chosen on CV, not test
    final = models()[best].fit(train.text, train.y)
    Path("models").mkdir(exist_ok=True)
    pickle.dump(final, open("models/spam_model.pkl", "wb"))

    # most spam-like words (logistic regression word features)
    lr = models()["Logistic Regression"].fit(df.text, df.y)
    union, clf = lr.steps[0][1], lr.steps[1][1]
    words = union.transformer_list[0][1].get_feature_names_out()
    coef = clf.coef_[0][: len(words)]
    top_spam = [words[i] for i in np.argsort(coef)[-15:][::-1]]
    top_ham = [words[i] for i in np.argsort(coef)[:15]]

    # error examples from the chosen model
    p = test_probs[best]
    errs = test.assign(p=p)
    fp = errs[(errs.y == 0) & (errs.p >= 0.5)].text.head(3).tolist()
    fn = errs[(errs.y == 1) & (errs.p < 0.5)].text.head(3).tolist()

    report = {"data": {"messages": len(df), "spam": int(df.y.sum()), "duplicates_removed": 403,
                       "train": len(train), "test": len(test)},
              "chosen_model": best, "results": results, "top_spam_words": [pretty(w) for w in top_spam], "top_ham_words": [pretty(w) for w in top_ham],
              "examples": {"false_alarms": fp, "missed_spam": fn}}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))

    # figures
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, pr in test_probs.items():
        pr_, rc_, _ = precision_recall_curve(test.y, pr)
        ax.plot(rc_, pr_, lw=2.4 if name == best else 1.3, color=SPAM if name == best else None,
                label=f"{name} (AP {results[name]['test']['pr_auc']:.3f})")
    ax.set(xlabel="Recall (spam caught)", ylabel="Precision (flagged that really is spam)",
           title="Precision–recall on the held-out test set", xlim=(0.5, 1.01), ylim=(0.5, 1.01))
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "precision_recall.png", dpi=150); plt.close(fig)

    cm = confusion_matrix(test.y, (p >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Oranges")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=15, color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
           xlabel="Predicted", ylabel="Actual", title=f"{best}: test set")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, ws, c, ttl, sign in [(axes[0], top_spam, SPAM, "Words that push towards SPAM", 1),
                                 (axes[1], top_ham, HAM, "Words that push towards HAM", -1)]:
        vals = [abs(coef[list(words).index(w)]) for w in ws][::-1]
        ax.barh([pretty(w) for w in ws][::-1], vals, color=c)
        ax.set(title=ttl, xlabel="Logistic regression weight")
    fig.tight_layout(); fig.savefig(FIG / "top_words.png", dpi=150); plt.close(fig)
    print("chosen:", best)


if __name__ == "__main__":
    main()
