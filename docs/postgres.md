# Local Services

The quickest supported service setup is:

```bash
docker compose up -d postgres redis
```

Default local DSNs:

```bash
export DATABASE_URL=postgresql://invoice_router:invoice_router@localhost:5432/invoice_router
export ANALYSIS_DATABASE_URL=postgresql://invoice_router:invoice_router@localhost:5432/invoice_router
export REDIS_URL=redis://localhost:6379/0
```

Optional runtime paths:

```bash
export DATASET_ROOT=$PWD/samples
export OUTPUT_DIR=$PWD/.local/output
export TEMP_DIR=$PWD/.local/temp
```

Benchmark commands derive isolated Postgres databases from `DATABASE_URL`, for example `invoice_router_benchmark_demo_native_pdf_heuristic`.
