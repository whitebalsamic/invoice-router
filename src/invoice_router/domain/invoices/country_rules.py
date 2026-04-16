import re
from typing import Dict, List, Optional, Tuple

SUPPORTED_COUNTRIES = {"US", "CA", "MX"}

_CANADIAN_POSTAL_RE = re.compile(
    r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b", re.IGNORECASE
)
_US_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_MX_POSTAL_RE = re.compile(
    r"\b(?:c\.?\s*p\.?|codigo postal|c[oó]digo postal)\s*:?\s*\d{5}\b", re.IGNORECASE
)
_MX_RFC_RE = re.compile(r"\b[A-Z&Ñ]{3,4}\d{6}[A-Z0-9]{3}\b", re.IGNORECASE)

_CA_PROVINCE_CODES = {
    "ab",
    "bc",
    "mb",
    "nb",
    "nl",
    "ns",
    "nt",
    "nu",
    "on",
    "pe",
    "qc",
    "sk",
    "yt",
}
_CA_PROVINCE_NAMES = {
    "alberta",
    "british columbia",
    "manitoba",
    "new brunswick",
    "newfoundland and labrador",
    "nova scotia",
    "northwest territories",
    "nunavut",
    "ontario",
    "prince edward island",
    "quebec",
    "saskatchewan",
    "yukon",
}
_US_STATE_CODES = {
    "al",
    "ak",
    "az",
    "ar",
    "ca",
    "co",
    "ct",
    "de",
    "fl",
    "ga",
    "hi",
    "id",
    "il",
    "in",
    "ia",
    "ks",
    "ky",
    "la",
    "me",
    "md",
    "ma",
    "mi",
    "mn",
    "ms",
    "mo",
    "mt",
    "ne",
    "nv",
    "nh",
    "nj",
    "nm",
    "ny",
    "nc",
    "nd",
    "oh",
    "ok",
    "or",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx",
    "ut",
    "vt",
    "va",
    "wa",
    "wv",
    "wi",
    "wy",
    "dc",
}
_US_STATE_NAMES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "district of columbia",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
}
_MX_STATE_NAMES = {
    "aguascalientes",
    "baja california",
    "campeche",
    "chiapas",
    "chihuahua",
    "coahuila",
    "colima",
    "durango",
    "guanajuato",
    "guerrero",
    "hidalgo",
    "jalisco",
    "mexico city",
    "michoacan",
    "morelos",
    "nayarit",
    "nuevo leon",
    "oaxaca",
    "puebla",
    "queretaro",
    "quintana roo",
    "san luis potosi",
    "sinaloa",
    "sonora",
    "tabasco",
    "tamaulipas",
    "tlaxcala",
    "veracruz",
    "yucatan",
    "zacatecas",
}


def _has_marker(text: str, marker: str) -> bool:
    if re.fullmatch(r"[a-z]{2,4}", marker):
        return re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", text) is not None
    return marker in text


def _score_markers(lowered: str) -> Dict[str, int]:
    scores = {"US": 0, "CA": 0, "MX": 0}

    if any(
        _has_marker(lowered, marker)
        for marker in ("cad", "canada", "gst", "hst", "pst", "tps", "tvq", "postal code")
    ):
        scores["CA"] += 3
    if any(
        _has_marker(lowered, marker) for marker in ("usd", "united states", "zip code", "sales tax")
    ):
        scores["US"] += 3
    if any(
        _has_marker(lowered, marker)
        for marker in ("mxn", "mexico", "rfc", "iva", "c.p.", "codigo postal", "código postal")
    ):
        scores["MX"] += 3

    if _CANADIAN_POSTAL_RE.search(lowered):
        scores["CA"] += 4
    if _MX_POSTAL_RE.search(lowered) or _MX_RFC_RE.search(lowered):
        scores["MX"] += 4
    if _US_ZIP_RE.search(lowered):
        scores["US"] += 1

    if any(name in lowered for name in _CA_PROVINCE_NAMES):
        scores["CA"] += 2
    if any(name in lowered for name in _US_STATE_NAMES) and _US_ZIP_RE.search(lowered):
        scores["US"] += 3
    if any(name in lowered for name in _MX_STATE_NAMES):
        scores["MX"] += 2

    tokens = re.findall(r"[a-z]{2,}", lowered)
    if any(code in tokens for code in _CA_PROVINCE_CODES) and _CANADIAN_POSTAL_RE.search(lowered):
        scores["CA"] += 3
    if any(code in tokens for code in _US_STATE_CODES) and _US_ZIP_RE.search(lowered):
        scores["US"] += 3

    return scores


def infer_country_and_currency(tokens: List[str]) -> Tuple[Optional[str], Optional[str]]:
    joined = " ".join(tokens)
    lowered = joined.lower()
    scores = _score_markers(lowered)

    explicit_currency = None
    if _has_marker(lowered, "cad"):
        explicit_currency = "CAD"
    elif _has_marker(lowered, "mxn"):
        explicit_currency = "MXN"
    elif _has_marker(lowered, "usd"):
        explicit_currency = "USD"

    best_country = None
    best_score = 0
    for country_code, score in scores.items():
        if score > best_score:
            best_country = country_code
            best_score = score

    if best_country and best_score >= 3:
        return best_country, explicit_currency or expected_currency_for_country(best_country)
    if explicit_currency:
        return None, explicit_currency
    if "$" in joined:
        return None, "USD"
    return None, None


def expected_currency_for_country(country_code: Optional[str]) -> Optional[str]:
    if country_code == "US":
        return "USD"
    if country_code == "CA":
        return "CAD"
    if country_code == "MX":
        return "MXN"
    return None


def validate_country_currency(
    country_code: Optional[str], currency_code: Optional[str]
) -> List[str]:
    errors: List[str] = []
    if country_code and country_code not in SUPPORTED_COUNTRIES:
        errors.append(f"Unsupported country code: {country_code}")
        return errors

    expected_currency = expected_currency_for_country(country_code)
    if expected_currency and currency_code and currency_code != expected_currency:
        errors.append(
            f"Currency mismatch for country {country_code}: expected {expected_currency}, got {currency_code}"
        )
    return errors


def normalize_party_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned or None
