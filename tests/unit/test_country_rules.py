from invoice_router.domain.invoices.country_rules import (
    infer_country_and_currency,
    validate_country_currency,
)


def test_infer_country_and_currency_for_canada():
    country, currency = infer_country_and_currency(["GST", "Postal Code", "CAD"])

    assert country == "CA"
    assert currency == "CAD"


def test_infer_country_and_currency_for_us_address_markers():
    country, currency = infer_country_and_currency(["Austin", "TX", "78701", "ZIP"])

    assert country == "US"
    assert currency == "USD"


def test_infer_country_and_currency_for_mexico_rfc_and_postal_code():
    country, currency = infer_country_and_currency(["RFC", "XAXX010101000", "C.P. 06600"])

    assert country == "MX"
    assert currency == "MXN"


def test_infer_country_and_currency_ignores_currency_substrings_inside_words():
    country, currency = infer_country_and_currency(["Escada", "Taylor", "RI", "87544"])

    assert country == "US"
    assert currency == "USD"


def test_validate_country_currency_detects_mismatch():
    errors = validate_country_currency("MX", "USD")

    assert errors == ["Currency mismatch for country MX: expected MXN, got USD"]
