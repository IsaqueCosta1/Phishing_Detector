"""Configuração central: caminhos, semente e premissas de negócio."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
RAW_CSV = DATA_RAW / "Phishing_Legitimate_full.csv"

RANDOM_STATE = 42
TARGET = "CLASS_LABEL"      # 1 = phishing, 0 = legítimo
ID_COLUMN = "id"


@dataclass(frozen=True)
class CostMatrix:
    """Custo relativo dos dois erros.

    Premissa do projeto: um phishing não detectado custa 10x um site
    legítimo bloqueado. Decisão de negócio, não fato empírico, mude
    aqui e o limiar ótimo se reajusta sozinho.
    """
    false_negative: float = 10.0
    false_positive: float = 1.0

    def total(self, fp: int, fn: int) -> float:
        return fp * self.false_positive + fn * self.false_negative


COSTS = CostMatrix()


URL_FEATURES: tuple[str, ...] = (
    "NumDots", "SubdomainLevel", "PathLevel", "UrlLength", "NumDash",
    "NumDashInHostname", "AtSymbol", "TildeSymbol", "NumUnderscore",
    "NumPercent", "NumQueryComponents", "NumAmpersand", "NumHash",
    "NumNumericChars", "NoHttps", "RandomString", "IpAddress",
    "DomainInSubdomains", "DomainInPaths", "HttpsInHostname",
    "HostnameLength", "PathLength", "QueryLength", "DoubleSlashInPath",
    "NumSensitiveWords", "EmbeddedBrandName", "SubdomainLevelRT",
    "UrlLengthRT",
)

HTML_FEATURES: tuple[str, ...] = (
    "PctExtHyperlinks", "PctExtResourceUrls", "ExtFavicon", "InsecureForms",
    "RelativeFormAction", "ExtFormAction", "AbnormalFormAction",
    "PctNullSelfRedirectHyperlinks", "FrequentDomainNameMismatch",
    "FakeLinkInStatusBar", "RightClickDisabled", "PopUpWindow",
    "SubmitInfoToEmail", "IframeOrFrame", "MissingTitle", "ImagesOnlyInForm",
    "PctExtResourceUrlsRT", "AbnormalExtFormActionR", "ExtMetaScriptLinkRT",
    "PctExtNullSelfRedirectHyperlinksRT",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "url_only": URL_FEATURES,
    "html_only": HTML_FEATURES,
    "full": URL_FEATURES + HTML_FEATURES,
}