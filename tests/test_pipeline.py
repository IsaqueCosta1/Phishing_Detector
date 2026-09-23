import numpy as np
import pandas as pd
import pytest

from phishing import config, data


@pytest.fixture
def sintetico() -> pd.DataFrame:
    """DataFrame com as mesmas patologias do dataset real:
    coluna constante e linhas duplicadas."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "UrlLength": rng.integers(10, 200, n),
        "NumDots": rng.integers(1, 8, n),
        "HttpsInHostname": 0,                    # constante, como no real
        "PctExtHyperlinks": rng.random(n),
        config.TARGET: rng.integers(0, 2, n),
    })
    return pd.concat([df, df.iloc[:20]], ignore_index=True)   # 20 duplicatas


def test_clean_remove_duplicatas_e_constantes(sintetico):
    limpo, audit = data.clean(sintetico)
    assert audit.n_duplicates_removed == 20
    assert "HttpsInHostname" in audit.constant_columns_removed
    assert config.ID_COLUMN not in limpo.columns


def test_nenhuma_linha_de_teste_duplica_linha_de_treino(sintetico):
    limpo, _ = data.clean(sintetico)
    X_tr, X_te, _, _ = data.split(limpo)
    assert len(pd.merge(X_tr, X_te, how="inner")) == 0