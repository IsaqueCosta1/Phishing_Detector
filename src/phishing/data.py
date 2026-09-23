"""Carga, auditoria, limpeza e particionamento."""

from dataclasses import dataclass, asdict
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from phishing import config

# Auditoria

@dataclass
class DataAudit:
    """O que foi encontrado e removido. Vira artefato em reports/."""
    n_rows_raw: int
    n_missing: int
    n_duplicates_removed: int
    constant_columns_removed: list[str]
    n_rows_clean: int
    class_balance: dict[str, int]

# Carga

def load_raw(path=None) -> pd.DataFrame:
    path = path or config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"CSV não encontrado em {path}. Baixe do Kaggle e coloque em data/raw/."
        )
    return pd.read_csv(path)

# Limpeza

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, DataAudit]:
    """Remove id, colunas constantes e duplicatas — nesta ordem."""
    out = df.drop(columns=[config.ID_COLUMN], errors="ignore")

    constant_cols = [c for c in out.columns if out[c].nunique(dropna=False) <= 1]
    out = out.drop(columns=constant_cols)

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)

    audit = DataAudit(
        n_rows_raw=len(df),
        n_missing=int(df.isna().sum().sum()),
        n_duplicates_removed=before - len(out),
        constant_columns_removed=constant_cols,
        n_rows_clean=len(out),
        class_balance={str(k): int(v)
                       for k, v in out[config.TARGET].value_counts().items()},
    )
    return out, audit

# Particionamento

def split(df: pd.DataFrame,
          test_size: float = 0.20, # 20% para teste
          random_state: int = config.RANDOM_STATE):
    """Hold-out estratificado. O teste só é tocado uma vez, no final."""
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )