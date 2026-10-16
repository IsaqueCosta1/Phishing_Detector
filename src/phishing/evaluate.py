"""Escolha do limiar de decisão pelo custo e métricas finais."""

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)

from phishing import config


def confusion_at(y, scores, threshold, costs=config.COSTS) -> dict:
    """FP, FN e custo total ao bloquear tudo com score >= threshold."""
    y = np.asarray(y)
    pred = np.asarray(scores) >= threshold  # True = bloqueia
    fp = int(np.sum(pred & (y == 0)))       # legítimo bloqueado
    fn = int(np.sum(~pred & (y == 1)))      # phishing que passou
    return {"limiar": float(threshold), "fp": fp, "fn": fn,
            "custo": costs.total(fp=fp, fn=fn)}


def cost_curve(y, scores, costs=config.COSTS) -> pd.DataFrame:
    """Custo total (e FP, FN) para cada limiar entre 0 e 1."""
    return pd.DataFrame([confusion_at(y, scores, t, costs)
                         for t in np.linspace(0, 1, 1001)])


def best_threshold(y, scores, costs=config.COSTS) -> float:
    """Limiar de menor custo. Em empate, o mais alto (bloqueia menos)."""
    curva = cost_curve(y, scores, costs)
    empatados = curva.loc[curva["custo"] == curva["custo"].min(), "limiar"]
    return float(empatados.max())


def report(y, scores, threshold, costs=config.COSTS) -> dict:
    """Métricas num limiar já fixado. Não escolhe nada a partir de y.

    ROC-AUC e AP olham só a ordenação dos scores (independem do limiar);
    recall, precisão e custo dependem da decisão tomada no limiar.
    """
    y = np.asarray(y)
    pred = np.asarray(scores) >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        **confusion_at(y, scores, threshold, costs),
    }
