import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishing import interpret


def test_permutacao_acha_a_feature_que_decide_e_ignora_ruido():
    # Só "sinal" define o rótulo; "ruido" é aleatório puro.
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"ruido": rng.random(400), "sinal": rng.random(400)})
    y = pd.Series((X["sinal"] > 0.5).astype(int))
    pipe = Pipeline([("model", LogisticRegression())])

    imp = interpret.permutation_importance_cv(pipe, X, y, n_splits=3, n_repeats=3)

    assert imp.loc[0, "feature"] == "sinal"
    assert imp.set_index("feature").loc["ruido", "queda_media"] < 0.01
