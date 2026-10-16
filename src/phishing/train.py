import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from phishing import config, data, evaluate, models
from phishing.predict import Detector

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


def _cv(n_splits: int) -> StratifiedKFold:
    """Mesmos folds em toda etapa, para as comparações serem justas."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=config.RANDOM_STATE)


def _avaliar(pipe, X, y, cv) -> dict[str, float]:
    """Roda a CV e resume cada métrica em média e desvio."""
    scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING,
                            n_jobs=-1, error_score="raise")
    resumo = {}
    for metrica in SCORING:
        resumo[f"{metrica}_mean"] = float(np.mean(scores[f"test_{metrica}"]))
        resumo[f"{metrica}_std"] = float(np.std(scores[f"test_{metrica}"]))
    return resumo


def compare_models(X_train, y_train, n_splits: int = 5) -> pd.DataFrame:
    """Compara candidatos por CV estratificada. Só toca no treino."""
    cv = _cv(n_splits)
    linhas = []
    for nome, pipe in models.get_candidates().items():
        t0 = time.perf_counter()
        linha = {"model": nome, **_avaliar(pipe, X_train, y_train, cv)}
        linha["fit_time_s"] = round(time.perf_counter() - t0, 2)
        linhas.append(linha)
        print(f"  {nome:<24} ROC-AUC={linha['roc_auc_mean']:.4f} "
              f"(±{linha['roc_auc_std']:.4f})  [{linha['fit_time_s']}s]")
    return (pd.DataFrame(linhas)
              .sort_values("roc_auc_mean", ascending=False)
              .reset_index(drop=True))


def ablation(X_train, y_train, model_name: str = config.FINAL_MODEL,
             n_splits: int = 5) -> pd.DataFrame:
    """Mesmo modelo, conjuntos de features diferentes. Só toca no treino."""
    cv = _cv(n_splits)
    linhas = []
    for nome_conjunto, features in config.FEATURE_SETS.items():
        pipe = models.get_candidates()[model_name]
        X = X_train[list(features)]
        linhas.append({"feature_set": nome_conjunto,
                       "n_features": len(features),
                       **_avaliar(pipe, X, y_train, cv)})
    return pd.DataFrame(linhas)


def oof_scores(X_train, y_train, model_name: str = config.FINAL_MODEL,
               n_splits: int = 5) -> np.ndarray:
    """Score de cada linha do treino dado por um modelo que não a viu.

    É o substituto honesto do teste para escolher o limiar: scores do
    modelo no próprio treino ficam colados em 0 e 1 e escondem os erros.
    """
    pipe = models.get_candidates()[model_name]
    return cross_val_predict(pipe, X_train, y_train, cv=_cv(n_splits),
                             method="predict_proba", n_jobs=-1)[:, 1]


def final_evaluation(X_train, y_train, X_test, y_test, threshold: float,
                     model_name: str = config.FINAL_MODEL):
    """Treina no treino inteiro e mede no teste, com o limiar já escolhido.

    É a única função do projeto que toca no teste. Devolve o modelo
    treinado e um dict {nome: métricas} para o limiar escolhido, o padrão
    0,5 e o 'oráculo' (melhor limiar olhando o teste — só diagnóstico,
    mede o quanto o limiar do OOF se afasta do ideal; nunca é usado).
    """
    pipe = models.get_candidates()[model_name].fit(X_train, y_train)
    scores = pipe.predict_proba(X_test)[:, 1]
    oraculo = evaluate.best_threshold(y_test, scores)
    resultados = {
        "escolhido": evaluate.report(y_test, scores, threshold),
        "padrao_0.5": evaluate.report(y_test, scores, 0.5),
        "oraculo_teste": evaluate.report(y_test, scores, oraculo),
    }
    return pipe, resultados


def save_report(conteudo: dict, nome: str = "metrics.json") -> None:
    config.REPORTS.mkdir(exist_ok=True)
    caminho = config.REPORTS / nome
    caminho.write_text(json.dumps(conteudo, indent=2, ensure_ascii=False))
    print(f"\nRelatório salvo em {caminho.relative_to(config.ROOT)}")


def main():
    df_limpo, audit = data.clean(data.load_raw())
    X_tr, X_te, y_tr, y_te = data.split(df_limpo)
    print(f"Treino: {len(X_tr)} linhas | Teste: {len(X_te)} linhas (reservado)\n")

    ranking = compare_models(X_tr, y_tr)
    print("\nRanking (média ± desvio em 5 folds):")
    colunas = ["model", "roc_auc_mean", "roc_auc_std",
               "average_precision_mean", "recall_mean", "precision_mean"]
    print(ranking[colunas].to_string(index=False, float_format="%.4f"))

    print(f"\nAblação ({config.FINAL_MODEL}):")
    abl = ablation(X_tr, y_tr)
    colunas_abl = ["feature_set", "n_features", "roc_auc_mean",
                   "roc_auc_std", "average_precision_mean", "recall_mean"]
    print(abl[colunas_abl].to_string(index=False, float_format="%.4f"))

    razao = config.COSTS.false_negative / config.COSTS.false_positive
    print(f"\nLimiar de decisão (out-of-fold, FN = {razao:g}x FP):")
    scores = oof_scores(X_tr, y_tr)
    limiar = evaluate.best_threshold(y_tr, scores)
    comparacao = pd.DataFrame([evaluate.confusion_at(y_tr, scores, 0.5),
                               evaluate.confusion_at(y_tr, scores, limiar)],
                              index=["padrão", "ótimo"])
    print(comparacao.to_string(float_format="%.3f"))

    print("\nAvaliação final no teste (única vez que o teste é usado):")
    modelo, teste = final_evaluation(X_tr, y_tr, X_te, y_te, limiar)
    print(pd.DataFrame(teste).T.to_string(float_format="%.4f"))

    # Salva exatamente o modelo avaliado acima, com o limiar escolhido no OOF.
    detector = Detector(pipeline=modelo, threshold=limiar,
                        features=list(X_tr.columns),
                        metadata={"modelo": config.FINAL_MODEL,
                                  "n_linhas_treino": len(X_tr),
                                  "custos": asdict(config.COSTS)})
    caminho = detector.save()
    print(f"\nDetector salvo em {caminho.relative_to(config.ROOT)}")

    save_report({
        "auditoria": asdict(audit),
        "modelo_final": config.FINAL_MODEL,
        "custos": asdict(config.COSTS),
        "cv_ranking": ranking.to_dict(orient="records"),
        "ablacao": abl.to_dict(orient="records"),
        "limiar": {"escolhido_oof": limiar,
                   "oof": comparacao.to_dict(orient="index")},
        "teste": teste,
    })


if __name__ == "__main__":
    main()
