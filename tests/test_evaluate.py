import numpy as np
from phishing import config, evaluate


def test_limiar_desce_quando_falso_negativo_fica_mais_caro():
    # dados pequenos, inventados, em que você sabe a resposta
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.3, 0.8, 0.9])

    simetrico = evaluate.best_threshold(y, scores, config.CostMatrix(1.0, 1.0))
    assimetrico = evaluate.best_threshold(y, scores, config.CostMatrix(10.0, 1.0))

    # FN caro: o limiar desce até pegar o phishing de score 0.3,
    # mesmo bloqueando o legítimo de 0.4 no caminho.
    assert assimetrico < simetrico
    assert assimetrico <= 0.3


def test_confusion_at_conta_erros_e_custo():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.3, 0.8, 0.9])

    # Limiar 0.5: bloqueia 0.6 (legítimo -> FP) e deixa passar 0.3 (FN).
    r = evaluate.confusion_at(y, scores, 0.5, config.CostMatrix(10.0, 1.0))
    assert (r["fp"], r["fn"]) == (1, 1)
    assert r["custo"] == 11.0
