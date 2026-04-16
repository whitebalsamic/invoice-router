from typing import List, Tuple

from ...config import AppConfig
from ...models import ProviderMatch


def resolve_provider(
    ocr_results_per_page: List[List[Tuple[str, int, int, int, int]]], config: AppConfig
) -> ProviderMatch | None:
    """Resolve a likely provider from OCR text using configured aliases."""
    if not config.provider_resolution.providers:
        return None

    tokens = []
    for page in ocr_results_per_page:
        tokens.extend(text.lower() for text, *_ in page if text.strip())

    joined_text = " ".join(tokens)
    best_match: ProviderMatch | None = None

    for provider_name, entry in config.provider_resolution.providers.items():
        aliases = [provider_name, *entry.aliases]
        matched_on = []
        best_alias_score = 0.0
        for alias in aliases:
            alias_norm = alias.strip().lower()
            if not alias_norm:
                continue
            if alias_norm in joined_text:
                score = 1.0
            else:
                alias_tokens = [token for token in alias_norm.split() if len(token) > 2]
                if not alias_tokens:
                    continue
                hits = sum(1 for token in alias_tokens if token in joined_text)
                score = hits / len(alias_tokens)
            if score > 0:
                matched_on.append(alias)
            best_alias_score = max(best_alias_score, score)

        if best_alias_score >= config.provider_resolution.minimum_confidence:
            candidate = ProviderMatch(
                provider_name=provider_name,
                confidence=best_alias_score,
                matched_on=matched_on,
                country_code=entry.country_code,
            )
            if best_match is None or candidate.confidence > best_match.confidence:
                best_match = candidate

    return best_match
