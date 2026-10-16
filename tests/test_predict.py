import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from phishing.predict import Detector, main


@pytest.fixture
def detector_e_dados():
    """Detector minúsculo treinado em dados sintéticos com 2 features."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(200), "b": rng.random(200)})
    y = (X["a"] > 0.5).astype(int)
    det = Detector(pipeline=LogisticRegression().fit(X, y),
                   threshold=0.3, features=["a", "b"])
    return det, X


def test_salvar_e_carregar_preserva_scores_e_limiar(detector_e_dados, tmp_path):
    det, X = detector_e_dados
    det.save(tmp_path / "det.joblib")
    carregado = Detector.load(tmp_path / "det.joblib")

    assert carregado.threshold == 0.3
    np.testing.assert_allclose(carregado.score(X), det.score(X))


def test_decisao_usa_o_limiar_do_detector_e_nao_05(detector_e_dados):
    det, X = detector_e_dados
    r = det.predict(X)
    assert (r["bloquear"] == (r["score"] >= 0.3)).all()
    assert (r["bloquear"] & (r["score"] < 0.5)).any()   # 0,5 não bloquearia


def test_ignora_colunas_extras_e_ordem(detector_e_dados):
    det, X = detector_e_dados
    baguncado = X[["b", "a"]].assign(id=range(len(X)), CLASS_LABEL=0)
    np.testing.assert_allclose(det.score(baguncado), det.score(X))


def test_feature_faltando_gera_erro_claro(detector_e_dados):
    det, X = detector_e_dados
    with pytest.raises(ValueError, match="Faltam 1 features.*b"):
        det.score(X[["a"]])


def test_cli_arquivo_inexistente_sai_sem_traceback(capsys):
    with pytest.raises(SystemExit) as saida:
        main(["nao_existe.csv"])
    assert saida.value.code != 0
    assert "arquivo não encontrado" in capsys.readouterr().err
