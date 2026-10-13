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
