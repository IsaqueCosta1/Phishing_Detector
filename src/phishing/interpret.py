"""Interpretabilidade: o que o modelo usa (permutação) e como usa (SHAP)."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from phishing import config, data


def permutation_importance_cv(pipe, X, y, n_splits: int = 5, n_repeats: int = 5,
                              scoring: str = "roc_auc") -> pd.DataFrame:
    """Queda de desempenho ao embaralhar cada feature, medida na validação.

    Em cada fold, treina no resto e embaralha a feature no fold que o
    modelo não viu: mede o quanto ele *depende* dela para generalizar.
    Medida no treino, mostraria o quanto ele a decorou. Só toca no treino.

    Atenção: features correlacionadas se cobrem — embaralhar uma dói pouco
    se a gêmea continua intacta. Importância baixa não prova inutilidade.
    """
    quedas = []
    for idx_tr, idx_va in data.cv_folds(n_splits).split(X, y):
        modelo = clone(pipe).fit(X.iloc[idx_tr], y.iloc[idx_tr])
        r = permutation_importance(modelo, X.iloc[idx_va], y.iloc[idx_va],
                                   scoring=scoring, n_repeats=n_repeats,
                                   random_state=config.RANDOM_STATE, n_jobs=-1)
        quedas.append(r.importances)              # (n_features, n_repeats)
    todas = np.hstack(quedas)                     # (n_features, folds × repeats)
    return (pd.DataFrame({"feature": X.columns,
                          "queda_media": todas.mean(axis=1),
                          "queda_std": todas.std(axis=1)})
              .sort_values("queda_media", ascending=False)
              .reset_index(drop=True))


def shap_explanation(pipe, X, n_amostras: int = 1000):
    """Valores SHAP do modelo final numa amostra de X.

    Cada valor é a contribuição da feature para o score daquele site, em
    log-odds: base + soma das contribuições = score do modelo. Pensado para
    pipelines de árvore sem pré-processamento (o caso do FINAL_MODEL).
    """
    import shap  # dependência opcional, só da análise (extra [dev])

    amostra = X.sample(min(n_amostras, len(X)), random_state=config.RANDOM_STATE)
    return shap.TreeExplainer(pipe.named_steps["model"])(amostra)
