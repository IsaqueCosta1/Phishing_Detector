"""Escolha do limiar de decisão pelo custo."""

import numpy as np
import pandas as pd

from phishing import config


def cost_curve(y, scores, costs=config.COSTS) -> pd.DataFrame:
    """Custo total (e FP, FN) para cada limiar entre 0 e 1."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    linhas = []
    for t in np.linspace(0, 1, 1001):
        pred = scores >= t                 # True = bloqueia
        fp = int(np.sum(pred & (y == 0)))  # legítimo bloqueado
        fn = int(np.sum(~pred & (y == 1))) # phishing que passou
        linhas.append({"limiar": t, "fp": fp, "fn": fn,
                       "custo": costs.total(fp=fp, fn=fn)})
    return pd.DataFrame(linhas)


def best_threshold(y, scores, costs=config.COSTS) -> float:
    """Limiar de menor custo. Em empate, o mais alto (bloqueia menos)."""
    curva = cost_curve(y, scores, costs)
    empatados = curva.loc[curva["custo"] == curva["custo"].min(), "limiar"]
    return float(empatados.max())
