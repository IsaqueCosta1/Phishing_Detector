# Detector de Phishing

Classificador de páginas de phishing a partir de atributos da URL e do HTML,
com o **limiar de decisão escolhido pelo custo de negócio**: deixar passar um
phishing custa 10× mais do que bloquear um site legítimo.

| | Limiar padrão (0,5) | **Limiar escolhido (0,105)** |
|---|---|---|
| Phishings que passaram (teste, n = 1.917) | 8 | **0** |
| Sites legítimos bloqueados | 14 | 41 |
| Custo total (FN = 10, FP = 1) | 94 | **41 (−56%)** |

ROC-AUC de **0,998 ± 0,0006** na validação cruzada (o número conservador) e
0,9996 no teste. O limiar foi escolhido sem olhar o teste e ficou a 4 pontos
de custo do ótimo que se obteria olhando para ele.

---

## Dados

[*Phishing Dataset for Machine Learning: Feature Evaluation*](https://doi.org/10.17632/h3cgnj8hft.1)
(Tan, 2018, Mendeley Data, CC BY 4.0): 10.000 páginas rotuladas, 5.000 de
phishing e 5.000 legítimas, com 48 atributos numéricos já extraídos.

A auditoria ([notebook 01](notebooks/01_analise_exploratoria.ipynb))
encontrou dois defeitos estruturais:

- **419 linhas duplicadas**, quase todas de phishing (671 linhas envolvidas
  contra 5 legítimas, e um grupo com 17 cópias idênticas). A interpretação é
  que são *kits* de phishing: o mesmo template implantado em vários domínios.
  Elas são removidas **antes** da divisão treino/teste porque uma cópia de
  cada lado faria o modelo ser avaliado em linhas que já viu.
- **`HttpsInHostname` é constante** (sempre 0) e é removida.

Depois da limpeza ficam 9.581 linhas e 47 features, divididas em 80% treino
e 20% teste, de forma estratificada.

## Metodologia

O teste fica trancado até o final. Toda escolha (modelo, features e limiar) é
feita **só com o treino**, por validação cruzada estratificada em 5 folds.

1. **Comparação de modelos.** De um `DummyClassifier` (o piso) a gradient boosting.
2. **Ablação URL vs HTML.** O mesmo modelo treinado com cada conjunto de
   features, para medir o que cada fonte de informação contribui.
3. **Limiar pelo custo.** O limiar que minimiza `FP × 1 + FN × 10` é escolhido
   nos scores *out-of-fold* do treino: cada linha recebe o score de um modelo
   que não a viu. Scores do modelo no próprio treino ficam colados em 0 e 1 e
   esconderiam os erros.
4. **Avaliação final.** O modelo é treinado no treino inteiro e aplicado ao
   teste **uma única vez**, com o limiar já fixado.

## Resultados

Os detalhes estão no [notebook 02](notebooks/02_modelagem_e_avaliacao.ipynb)
e os números completos em [`reports/metrics.json`](reports/metrics.json).

**Modelos** (validação cruzada, 5 folds):

| Modelo | ROC-AUC | Average precision | Recall @0,5 |
|---|---|---|---|
| **HistGradientBoosting** | **0,9981 ± 0,0006** | 0,9981 | 0,984 |
| Random Forest | 0,9979 ± 0,0006 | 0,9979 | 0,980 |
| Regressão logística | 0,9824 ± 0,0028 | 0,9795 | 0,946 |
| Árvore de decisão (prof. 6) | 0,9753 ± 0,0026 | 0,9628 | 0,955 |
| Dummy (piso) | 0,5000 | 0,4786 | 0,000 |

Random Forest e gradient boosting empatam dentro do desvio. O desempate foi
operacional: o gradient boosting treina ~3× mais rápido e o modelo salvo tem
0,4 MB, contra 25 MB do Random Forest.

**Ablação** (mesmo modelo, conjuntos de features diferentes):

| Features | Nº | ROC-AUC | Recall @0,5 |
|---|---|---|---|
| Só URL | 27 | 0,967 | 0,911 |
| Só HTML | 20 | 0,994 | 0,977 |
| URL + HTML | 47 | **0,998** | **0,984** |

Só com a URL, cerca de 1 em cada 11 phishings passaria. O HTML é a fonte mais
informativa, mas as duas juntas erram menos do que qualquer uma sozinha.

**Interpretabilidade** (permutation importance medida nos folds de validação, e SHAP):

- As duas técnicas concordam: 9 das 10 features mais importantes coincidem
  (Spearman 0,96). O modelo se apoia sobretudo em **para onde a página
  aponta**: links externos, links nulos ("#") e domínio dos links.
- A feature mais importante, `PctExtHyperlinks`, tem uma relação **em U** com o
  phishing: 86% das páginas sem nenhum link externo são phishing (kits
  autocontidos), assim como 96% das que têm mais de 80% de links externos
  (clones de marca). Os sites legítimos ficam no meio. A correlação linear da
  EDA (0,26) subestimava essa feature, e isso explica por que a regressão
  logística perde para as árvores.
- Nos 8 phishings que o limiar 0,5 deixaria passar no teste, o SHAP mostra o
  que os puxou para baixo: em 4 deles, o sinal mais forte de legitimidade foi
  um formulário enviado por HTTPS (`InsecureForms = 0`). São phishings que
  imitam o sinal clássico de segurança. O notebook 02 explica um deles.

**Limiar** (scores out-of-fold no treino): ao baixar o limiar de 0,5 para
0,105, os phishings perdidos caem de 60 para 25 e os bloqueios indevidos sobem
de 67 para 156. O custo total cai de 667 para 406 (−39%). No teste, a mesma
troca levou o custo de 94 para 41.

## Como rodar

Requer Python 3.10 ou superior. O dataset já está no repositório, em `data/raw/`.

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

python -m phishing.train     # compara modelos, ablação, limiar, teste final;
                             # salva models/phishing_detector.joblib e reports/metrics.json
python -m phishing.predict data/raw/Phishing_Legitimate_full.csv -o resultado.csv
pytest                       # 11 testes
```

O `predict` recebe um CSV com as 47 features (colunas extras são ignoradas) e
devolve o `score` e a decisão `bloquear` de cada linha, já com o limiar
escolhido. O limiar é salvo no mesmo arquivo do modelo, para que ninguém o
carregue e decida com o 0,5 padrão por engano.

## Estrutura

```
src/phishing/
├── config.py      # caminhos, matriz de custo, modelo final, grupos de features
├── data.py        # carga, auditoria, limpeza e divisão treino/teste
├── models.py      # candidatos (pipelines do scikit-learn)
├── train.py       # validação cruzada, ablação, scores out-of-fold, avaliação final
├── evaluate.py    # curva de custo, escolha do limiar, métricas
├── interpret.py   # permutation importance na CV e SHAP
└── predict.py     # Detector (modelo + limiar + features) e a CLI
notebooks/
├── 01_analise_exploratoria.ipynb
└── 02_modelagem_e_avaliacao.ipynb
tests/             # limpeza sem vazamento, limiar, interpretabilidade, detector, CLI
```

A lógica vive em `src/` e os notebooks só a chamam. Assim, o que os notebooks
mostram é exatamente o que o `train.py` executa.

## Limitações

- **Balanceamento artificial.** O dataset tem ~50% de phishing; no tráfego
  real, a proporção é uma fração de percentual. Com a mesma taxa de falsos
  positivos, a maioria dos bloqueios seria de sites legítimos, e o limiar
  precisaria ser recalibrado com a proporção real.
- **Dados de 2018.** Phishing muda rápido. Em produção, o modelo precisaria de
  monitoramento de *drift* e retreino.
- **Features prontas, não a URL crua.** O detector recebe os 47 atributos já
  calculados. Um sistema completo precisa de um extrator que os calcule a partir
  da URL e do HTML.
- **A matriz de custo é uma premissa**, não um valor medido. Ela fica isolada
  em `config.CostMatrix` para ser discutida; ao mudá-la, o limiar se reajusta
  sozinho.

## Próximos passos

- **Triagem em dois estágios:** a URL decide os casos claros antes de abrir a
  página, e o HTML só é baixado na zona cinzenta.
- **Modelo reduzido:** 26 das 47 features têm importância por permutação
  praticamente nula. Vale testar na CV se um modelo só com as restantes mantém
  o desempenho, o que deixaria o extrator de features mais barato.
- **Extrator de features** a partir da URL e do HTML, para classificar um link
  diretamente.
