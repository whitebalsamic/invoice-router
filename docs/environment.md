# Environment

Use Python `3.12` for native development.

## Native setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
brew install tesseract
```

## Config discovery

`invoice-router` resolves configuration in this order:

1. `--config path/to/config.yaml`
2. `INVOICE_ROUTER_CONFIG`
3. `./config.yaml`
4. packaged defaults at `invoice_router.config/defaults.yaml`

Environment variables still provide runtime values such as database, Redis, output, and dataset paths. Start from `.env.example`.

## Optional Paddle setup

```bash
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install paddleocr
```

If `ocr.table_engine` is switched to `paddle`, startup fails fast unless Paddle is installed and importable.
