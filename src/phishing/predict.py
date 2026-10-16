"""Detector empacotado (modelo + limiar + features) e CLI de predição.

Uso:
    python -m phishing.predict sites.csv [-o resultado.csv]
"""

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from phishing import config


@dataclass
class Detector:
    """Tudo o que é preciso para decidir, num objeto só.

    O limiar viaja junto com o modelo: um modelo salvo sem ele decide
    com o 0,5 do sklearn e volta a deixar passar phishing.
    """
    pipeline: object
    threshold: float
    features: list[str]
    metadata: dict = field(default_factory=dict)

    def _alinhar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mesmas colunas e ordem do treino; colunas extras são ignoradas."""
        faltando = [c for c in self.features if c not in df.columns]
        if faltando:
            raise ValueError(
                f"Faltam {len(faltando)} features no CSV: {', '.join(faltando[:5])}"
                + ("..." if len(faltando) > 5 else "")
            )
        return df[self.features]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidade de phishing de cada linha."""
        return self.pipeline.predict_proba(self._alinhar(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score e decisão (bloquear = score >= limiar) de cada linha."""
        s = self.score(df)
        return pd.DataFrame({"score": s, "bloquear": s >= self.threshold},
                            index=df.index)

    def save(self, path: Path = config.MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata["sklearn_version"] = sklearn.__version__
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path = config.MODEL_PATH) -> "Detector":
        """Carrega um detector salvo.

        joblib usa pickle por baixo, que executa código ao carregar:
        só abra arquivos que você mesmo gerou.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {path}. Rode antes: python -m phishing.train"
            )
        detector = joblib.load(path)
        versao = detector.metadata.get("sklearn_version")
        if versao != sklearn.__version__:
            warnings.warn(
                f"Modelo salvo com scikit-learn {versao}, rodando com "
                f"{sklearn.__version__}. Os scores podem mudar; retreine se puder."
            )
        return detector


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Classifica sites como phishing a partir das features de URL/HTML.")
    parser.add_argument("entrada", type=Path, help="CSV com as colunas de features")
    parser.add_argument("-o", "--saida", type=Path,
                        help="CSV de saída (entrada + score + bloquear)")
    parser.add_argument("--modelo", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args(argv)

    detector = Detector.load(args.modelo)
    df = pd.read_csv(args.entrada)
    resultado = df.join(detector.predict(df))

    n = int(resultado["bloquear"].sum())
    print(f"{len(df)} sites | {n} bloqueados ({n / len(df):.1%}) "
          f"| limiar {detector.threshold:.3f}")

    if args.saida:
        resultado.to_csv(args.saida, index=False)
        print(f"Resultado salvo em {args.saida}")
    else:
        colunas = [c for c in [config.ID_COLUMN] if c in df.columns]
        print(resultado[colunas + ["score", "bloquear"]].head(10)
              .to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
