from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from phishing.config import RANDOM_STATE


def _pipe(model, scale: bool = False) -> Pipeline:
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_candidates() -> dict[str, Pipeline]:
    return {
        "dummy": _pipe(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _pipe(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            scale=True,
        ),
        "decision_tree": _pipe(
            DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
        ),
        "random_forest": _pipe(
            RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE)
        ),
        "hist_gradient_boosting": _pipe(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
    }