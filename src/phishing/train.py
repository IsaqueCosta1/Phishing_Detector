import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from phishing import config, data, models

# O dummy nunca prevê a classe positiva, então a precisão fica indefinida (0/0).
# Declarar zero_division=0 torna a escolha explícita em vez de deixar o sklearn
# emitir warning a cada fold.
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
}


def compare_models(X_train, y_train, n_splits: int = 5) -> pd.DataFrame:
    """Compara candidatos por CV estratificada. Só toca no treino."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=config.RANDOM_STATE)
    linhas = []
    for nome, pipe in models.get_candidates().items():
        t0 = time.perf_counter()
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=SCORING, n_jobs=-1, error_score="raise")
        linha = {"model": nome, "fit_time_s": round(time.perf_counter() - t0, 2)}
        for metrica in SCORING:
            linha[f"{metrica}_mean"] = float(np.mean(scores[f"test_{metrica}"]))
            linha[f"{metrica}_std"] = float(np.std(scores[f"test_{metrica}"]))
        linhas.append(linha)
        print(f"  {nome:<24} ROC-AUC={linha['roc_auc_mean']:.4f} "
              f"(±{linha['roc_auc_std']:.4f})  [{linha['fit_time_s']}s]")

    return (pd.DataFrame(linhas)
              .sort_values("roc_auc_mean", ascending=False)
              .reset_index(drop=True))


def main():
    df_limpo, audit = data.clean(data.load_raw())
    X_tr, X_te, y_tr, y_te = data.split(df_limpo)
    print(f"Treino: {len(X_tr)} linhas | Teste: {len(X_te)} linhas (reservado)\n")

    ranking = compare_models(X_tr, y_tr)

    print("\nRanking (média ± desvio em 5 folds):")
    colunas = ["model", "roc_auc_mean", "roc_auc_std",
               "average_precision_mean", "recall_mean", "precision_mean"]
    print(ranking[colunas].to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()

