import json
from typing import Any, List, Mapping, Optional

from ...models import (
    FingerprintRecord,
    ProcessingDiagnostics,
    ProcessingResult,
    Provenance,
    Route,
    TemplateFamilyExample,
    TemplateFamilyRecord,
    TemplateFamilyVersion,
    TemplateStatus,
)
from .postgres import PostgresClient, is_postgres_target


class FingerprintDB:
    def __init__(self, db_target: str):
        if not is_postgres_target(db_target):
            raise ValueError(f"FingerprintDB requires a Postgres DATABASE_URL, got: {db_target}")
        self.db_target = db_target
        self.backend = "postgres"
        self._pg_client = PostgresClient(db_target)
        self._init_db()

    def _pg(self) -> PostgresClient:
        if self._pg_client is None:
            raise RuntimeError("Postgres client is not configured")
        return self._pg_client

    def _init_db(self):
        self._init_postgres_db()

    def _init_postgres_db(self):
        pg = self._pg()
        statements = [
            """
            CREATE TABLE IF NOT EXISTS fingerprints (
                fingerprint_hash TEXT PRIMARY KEY,
                visual_hashes TEXT NOT NULL,
                layout_template TEXT NOT NULL,
                page_fingerprints TEXT NOT NULL,
                template_family_id TEXT,
                provider_name TEXT,
                country_code TEXT,
                confidence DOUBLE PRECISION DEFAULT 0.0,
                apply_count INTEGER DEFAULT 0,
                reject_count INTEGER DEFAULT 0,
                gt_apply_count INTEGER DEFAULT 0,
                gt_reject_count INTEGER DEFAULT 0,
                gt_confidence DOUBLE PRECISION DEFAULT 0.0,
                status TEXT DEFAULT 'provisional',
                version TEXT DEFAULT 'v3',
                created_at TEXT NOT NULL,
                last_used TEXT,
                last_validated TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_fingerprints_status ON fingerprints(status)",
            """
            CREATE TABLE IF NOT EXISTS fingerprint_history (
                id BIGSERIAL PRIMARY KEY,
                fingerprint_hash TEXT NOT NULL,
                layout_template TEXT NOT NULL,
                confidence DOUBLE PRECISION,
                status TEXT,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                replaced_by TEXT,
                retirement_reason TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS processing_results (
                id BIGSERIAL PRIMARY KEY,
                invoice_path TEXT NOT NULL UNIQUE,
                fingerprint_hash TEXT,
                template_family_id TEXT,
                extracted_data TEXT,
                normalized_data TEXT,
                ground_truth TEXT,
                ground_truth_valid INTEGER,
                provenance TEXT,
                validation_passed INTEGER,
                route_used TEXT,
                attempted_route TEXT,
                diagnostics TEXT,
                image_quality_score DOUBLE PRECISION,
                template_status_at_time TEXT,
                processed_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS job_progress (
                job_id TEXT PRIMARY KEY,
                total INTEGER DEFAULT 0,
                done INTEGER DEFAULT 0,
                failed INTEGER DEFAULT 0,
                current_item TEXT,
                accuracy DOUBLE PRECISION,
                status TEXT DEFAULT 'PENDING',
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                errors TEXT DEFAULT '[]'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS template_families (
                template_family_id TEXT PRIMARY KEY,
                provider_name TEXT,
                country_code TEXT,
                document_family TEXT NOT NULL DEFAULT 'unknown',
                stable_anchor_regions TEXT NOT NULL DEFAULT '{}',
                anchor_summary TEXT NOT NULL DEFAULT '{}',
                page_role_expectations TEXT NOT NULL DEFAULT '[]',
                summary_area_anchors TEXT NOT NULL DEFAULT '{}',
                variable_region_masks TEXT NOT NULL DEFAULT '[]',
                extraction_profile TEXT NOT NULL DEFAULT '{}',
                confidence DOUBLE PRECISION DEFAULT 0.0,
                apply_count INTEGER DEFAULT 0,
                reject_count INTEGER DEFAULT 0,
                gt_apply_count INTEGER DEFAULT 0,
                gt_reject_count INTEGER DEFAULT 0,
                gt_confidence DOUBLE PRECISION DEFAULT 0.0,
                status TEXT DEFAULT 'provisional',
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS template_family_examples (
                id BIGSERIAL PRIMARY KEY,
                template_family_id TEXT NOT NULL,
                fingerprint_hash TEXT,
                invoice_path TEXT,
                example_metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS template_family_versions (
                id BIGSERIAL PRIMARY KEY,
                template_family_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                family_snapshot TEXT NOT NULL,
                change_reason TEXT,
                created_at TEXT NOT NULL
            )
            """,
            "ALTER TABLE processing_results ADD COLUMN IF NOT EXISTS attempted_route TEXT",
            "ALTER TABLE processing_results ADD COLUMN IF NOT EXISTS diagnostics TEXT",
            "ALTER TABLE processing_results ADD COLUMN IF NOT EXISTS normalized_data TEXT",
            "ALTER TABLE processing_results ADD COLUMN IF NOT EXISTS template_family_id TEXT",
            "ALTER TABLE fingerprints ADD COLUMN IF NOT EXISTS provider_name TEXT",
            "ALTER TABLE fingerprints ADD COLUMN IF NOT EXISTS country_code TEXT",
            "ALTER TABLE fingerprints ADD COLUMN IF NOT EXISTS template_family_id TEXT",
            "ALTER TABLE template_families ADD COLUMN IF NOT EXISTS anchor_summary TEXT DEFAULT '{}'",
            "ALTER TABLE template_families ADD COLUMN IF NOT EXISTS extraction_profile TEXT DEFAULT '{}'",
            "ALTER TABLE template_families ADD COLUMN IF NOT EXISTS apply_count INTEGER DEFAULT 0",
            "ALTER TABLE template_families ADD COLUMN IF NOT EXISTS reject_count INTEGER DEFAULT 0",
            "ALTER TABLE fingerprints ADD COLUMN IF NOT EXISTS gt_apply_count INTEGER DEFAULT 0",
            "ALTER TABLE fingerprints ADD COLUMN IF NOT EXISTS gt_reject_count INTEGER DEFAULT 0",
            "ALTER TABLE fingerprints ADD COLUMN IF NOT EXISTS gt_confidence DOUBLE PRECISION DEFAULT 0.0",
            "ALTER TABLE template_families ADD COLUMN IF NOT EXISTS gt_apply_count INTEGER DEFAULT 0",
            "ALTER TABLE template_families ADD COLUMN IF NOT EXISTS gt_reject_count INTEGER DEFAULT 0",
            "ALTER TABLE template_families ADD COLUMN IF NOT EXISTS gt_confidence DOUBLE PRECISION DEFAULT 0.0",
            "CREATE INDEX IF NOT EXISTS idx_fingerprints_template_family_id ON fingerprints(template_family_id)",
            "CREATE INDEX IF NOT EXISTS idx_template_family_examples_family_id ON template_family_examples(template_family_id)",
            "CREATE INDEX IF NOT EXISTS idx_template_family_versions_family_id ON template_family_versions(template_family_id)",
        ]
        for statement in statements:
            pg.execute(statement)

    def clear_processing_results(self, invoice_paths: Optional[List[str]] = None):
        if invoice_paths:
            placeholders = ",".join("?" * len(invoice_paths))
            self._pg().execute(
                f"DELETE FROM processing_results WHERE invoice_path IN ({placeholders})",
                invoice_paths,
            )
        else:
            self._pg().execute("DELETE FROM processing_results")

    def clear_fingerprints(self):
        self._pg().execute("DELETE FROM fingerprints")
        self._pg().execute("DELETE FROM fingerprint_history")

    def close(self):
        return None

    def store_result(self, result: ProcessingResult) -> int:
        params = (
            result.invoice_path,
            result.fingerprint_hash,
            result.template_family_id,
            json.dumps(result.extracted_data) if result.extracted_data is not None else None,
            json.dumps(result.normalized_data) if result.normalized_data is not None else None,
            json.dumps(result.ground_truth) if result.ground_truth is not None else None,
            1 if result.ground_truth is not None else 0,
            result.provenance.model_dump_json() if result.provenance else None,
            1
            if result.validation_passed is True
            else (0 if result.validation_passed is False else None),
            result.route_used.value if result.route_used else None,
            result.attempted_route.value if result.attempted_route else None,
            result.diagnostics.model_dump_json() if result.diagnostics else None,
            result.image_quality_score,
            result.template_status_at_time.value if result.template_status_at_time else None,
            result.processed_at,
        )

        row = self._pg().fetchone(
            """
            INSERT INTO processing_results (
                invoice_path, fingerprint_hash, template_family_id, extracted_data, normalized_data, ground_truth,
                ground_truth_valid, provenance, validation_passed, route_used,
                attempted_route, diagnostics, image_quality_score, template_status_at_time, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (invoice_path) DO UPDATE SET
                fingerprint_hash = EXCLUDED.fingerprint_hash,
                template_family_id = EXCLUDED.template_family_id,
                extracted_data = EXCLUDED.extracted_data,
                normalized_data = EXCLUDED.normalized_data,
                ground_truth = EXCLUDED.ground_truth,
                ground_truth_valid = EXCLUDED.ground_truth_valid,
                provenance = EXCLUDED.provenance,
                validation_passed = EXCLUDED.validation_passed,
                route_used = EXCLUDED.route_used,
                attempted_route = EXCLUDED.attempted_route,
                diagnostics = EXCLUDED.diagnostics,
                image_quality_score = EXCLUDED.image_quality_score,
                template_status_at_time = EXCLUDED.template_status_at_time,
                processed_at = EXCLUDED.processed_at
            RETURNING id
            """,
            params,
        )
        return int(row["id"]) if row else 0

    def get_result(self, invoice_path: str) -> Optional[ProcessingResult]:
        row = self._pg().fetchone(
            "SELECT * FROM processing_results WHERE invoice_path = ?", (invoice_path,)
        )
        return self._row_to_processing_result(row) if row else None

    def get_recent_results_for_family(
        self, template_family_id: str, limit: int = 10
    ) -> List[ProcessingResult]:
        rows = self._pg().fetchall(
            """
            SELECT * FROM processing_results
            WHERE template_family_id = ?
            ORDER BY processed_at DESC, id DESC
            LIMIT ?
            """,
            (template_family_id, limit),
        )
        return [self._row_to_processing_result(row) for row in rows]

    def get_results_for_fingerprints(
        self,
        fingerprint_hashes: List[str],
        *,
        limit: Optional[int] = None,
    ) -> List[ProcessingResult]:
        if not fingerprint_hashes:
            return []
        placeholders = ",".join("?" * len(fingerprint_hashes))
        params: List[Any] = list(fingerprint_hashes)
        sql = f"""
            SELECT * FROM processing_results
            WHERE fingerprint_hash IN ({placeholders})
            ORDER BY processed_at DESC, id DESC
        """
        if limit is not None:
            sql += "\nLIMIT ?"
            params.append(limit)
        rows = self._pg().fetchall(sql, params)
        return [self._row_to_processing_result(row) for row in rows]

    def get_failed_results(self) -> List[ProcessingResult]:
        rows = self._pg().fetchall("SELECT * FROM processing_results WHERE validation_passed = 0")
        return [self._row_to_processing_result(row) for row in rows]

    def get_failed_results_for_input_dir(
        self,
        input_dir: str,
        *,
        template_family_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProcessingResult]:
        normalized_input_dir = str(input_dir).rstrip("/")
        params: List[Any] = [f"{normalized_input_dir}/%"]
        sql = """
            SELECT * FROM processing_results
            WHERE validation_passed = 0
              AND invoice_path LIKE ?
        """
        if template_family_id is not None:
            sql += "\n  AND template_family_id = ?"
            params.append(template_family_id)
        sql += "\nORDER BY processed_at DESC, id DESC"
        if limit is not None:
            sql += "\nLIMIT ?"
            params.append(limit)
        rows = self._pg().fetchall(sql, params)
        return [self._row_to_processing_result(row) for row in rows]

    def get_recent_validation_passes(self, hash_val: str, limit: int) -> List[Optional[bool]]:
        rows = self._pg().fetchall(
            """
            SELECT validation_passed FROM processing_results
            WHERE fingerprint_hash = ? AND validation_passed IS NOT NULL
            ORDER BY processed_at DESC LIMIT ?
            """,
            (hash_val, limit),
        )
        return [bool(row["validation_passed"]) for row in rows]

    def _row_to_processing_result(self, row: Mapping[str, Any]) -> ProcessingResult:
        row_dict = dict(row)
        val_passed = None
        if row_dict.get("validation_passed") is not None:
            val_passed = bool(row_dict["validation_passed"])
        attempted_route = (
            Route(row_dict["attempted_route"]) if row_dict.get("attempted_route") else None
        )
        diagnostics = None
        if row_dict.get("diagnostics"):
            diagnostics = ProcessingDiagnostics.model_validate_json(row_dict["diagnostics"])

        return ProcessingResult(
            id=row_dict.get("id"),
            invoice_path=row_dict["invoice_path"],
            fingerprint_hash=row_dict.get("fingerprint_hash"),
            template_family_id=row_dict.get("template_family_id"),
            extracted_data=json.loads(row_dict["extracted_data"])
            if row_dict.get("extracted_data")
            else None,
            normalized_data=json.loads(row_dict["normalized_data"])
            if row_dict.get("normalized_data")
            else None,
            ground_truth=json.loads(row_dict["ground_truth"])
            if row_dict.get("ground_truth")
            else None,
            provenance=Provenance.model_validate_json(row_dict["provenance"])
            if row_dict.get("provenance")
            else None,
            validation_passed=val_passed,
            route_used=Route(row_dict["route_used"]) if row_dict.get("route_used") else None,
            attempted_route=attempted_route,
            diagnostics=diagnostics,
            image_quality_score=row_dict.get("image_quality_score"),
            template_status_at_time=TemplateStatus(row_dict["template_status_at_time"])
            if row_dict.get("template_status_at_time")
            else None,
            processed_at=row_dict["processed_at"],
        )

    def get_all_active_fingerprints(self) -> List[FingerprintRecord]:
        rows = self._pg().fetchall("SELECT * FROM fingerprints WHERE status != 'retired'")
        return [self._row_to_fingerprint_record(row) for row in rows]

    def get_fingerprints_for_family(self, template_family_id: str) -> List[FingerprintRecord]:
        rows = self._pg().fetchall(
            """
            SELECT * FROM fingerprints
            WHERE template_family_id = ? AND status != 'retired'
            ORDER BY confidence DESC, apply_count DESC, created_at
            """,
            (template_family_id,),
        )
        return [self._row_to_fingerprint_record(row) for row in rows]

    def store_fingerprint(
        self, record: FingerprintRecord, visual_hashes: List[str], page_fingerprints: List[dict]
    ):
        params = (
            record.hash,
            json.dumps(visual_hashes),
            json.dumps(record.layout_template),
            json.dumps(page_fingerprints),
            record.template_family_id,
            record.provider_name,
            record.country_code,
            record.confidence,
            record.apply_count,
            record.reject_count,
            record.gt_apply_count,
            record.gt_reject_count,
            record.gt_confidence,
            record.status.value,
            record.version,
            record.created_at,
            record.last_used,
        )

        self._pg().execute(
            """
            INSERT INTO fingerprints
            (fingerprint_hash, visual_hashes, layout_template, page_fingerprints, template_family_id, provider_name,
             country_code, confidence, apply_count, reject_count, gt_apply_count, gt_reject_count, gt_confidence,
             status, version, created_at, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (fingerprint_hash) DO UPDATE SET
                visual_hashes = EXCLUDED.visual_hashes,
                layout_template = EXCLUDED.layout_template,
                page_fingerprints = EXCLUDED.page_fingerprints,
                template_family_id = EXCLUDED.template_family_id,
                provider_name = EXCLUDED.provider_name,
                country_code = EXCLUDED.country_code,
                confidence = EXCLUDED.confidence,
                apply_count = EXCLUDED.apply_count,
                reject_count = EXCLUDED.reject_count,
                gt_apply_count = EXCLUDED.gt_apply_count,
                gt_reject_count = EXCLUDED.gt_reject_count,
                gt_confidence = EXCLUDED.gt_confidence,
                status = EXCLUDED.status,
                version = EXCLUDED.version,
                created_at = EXCLUDED.created_at,
                last_used = EXCLUDED.last_used
            """,
            params,
        )

    def store_template_family(self, record: TemplateFamilyRecord):
        params = (
            record.template_family_id,
            record.provider_name,
            record.country_code,
            record.document_family.value,
            json.dumps(record.stable_anchor_regions),
            json.dumps(record.anchor_summary),
            json.dumps([role.value for role in record.page_role_expectations]),
            json.dumps(record.summary_area_anchors),
            json.dumps(record.variable_region_masks),
            json.dumps(record.extraction_profile),
            record.confidence,
            record.apply_count,
            record.reject_count,
            record.gt_apply_count,
            record.gt_reject_count,
            record.gt_confidence,
            record.status.value,
            record.created_at,
            record.updated_at,
        )
        self._pg().execute(
            """
            INSERT INTO template_families (
                template_family_id, provider_name, country_code, document_family,
                stable_anchor_regions, anchor_summary, page_role_expectations, summary_area_anchors,
                variable_region_masks, extraction_profile, confidence, apply_count, reject_count,
                gt_apply_count, gt_reject_count, gt_confidence, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (template_family_id) DO UPDATE SET
                provider_name = EXCLUDED.provider_name,
                country_code = EXCLUDED.country_code,
                document_family = EXCLUDED.document_family,
                stable_anchor_regions = EXCLUDED.stable_anchor_regions,
                anchor_summary = EXCLUDED.anchor_summary,
                page_role_expectations = EXCLUDED.page_role_expectations,
                summary_area_anchors = EXCLUDED.summary_area_anchors,
                variable_region_masks = EXCLUDED.variable_region_masks,
                extraction_profile = EXCLUDED.extraction_profile,
                confidence = EXCLUDED.confidence,
                apply_count = EXCLUDED.apply_count,
                reject_count = EXCLUDED.reject_count,
                gt_apply_count = EXCLUDED.gt_apply_count,
                gt_reject_count = EXCLUDED.gt_reject_count,
                gt_confidence = EXCLUDED.gt_confidence,
                status = EXCLUDED.status,
                created_at = EXCLUDED.created_at,
                updated_at = EXCLUDED.updated_at
            """,
            params,
        )

    def get_template_family(self, template_family_id: str) -> Optional[TemplateFamilyRecord]:
        row = self._pg().fetchone(
            "SELECT * FROM template_families WHERE template_family_id = ?",
            (template_family_id,),
        )
        return self._row_to_template_family_record(row) if row else None

    def get_all_template_families(self) -> List[TemplateFamilyRecord]:
        rows = self._pg().fetchall("SELECT * FROM template_families ORDER BY template_family_id")
        return [self._row_to_template_family_record(row) for row in rows]

    def get_active_template_families(self) -> List[TemplateFamilyRecord]:
        rows = self._pg().fetchall(
            """
            SELECT * FROM template_families
            WHERE status != 'retired'
            ORDER BY template_family_id
            """
        )
        return [self._row_to_template_family_record(row) for row in rows]

    def update_template_family_lifecycle(
        self,
        template_family_id: str,
        *,
        confidence: float,
        apply_count: int,
        reject_count: int,
        gt_apply_count: Optional[int] = None,
        gt_reject_count: Optional[int] = None,
        gt_confidence: Optional[float] = None,
        status: TemplateStatus,
        updated_at: str,
    ):
        self._pg().execute(
            """
            UPDATE template_families
            SET confidence = ?, apply_count = ?, reject_count = ?,
                gt_apply_count = COALESCE(?, gt_apply_count),
                gt_reject_count = COALESCE(?, gt_reject_count),
                gt_confidence = COALESCE(?, gt_confidence),
                status = ?, updated_at = ?
            WHERE template_family_id = ?
            """,
            (
                confidence,
                apply_count,
                reject_count,
                gt_apply_count,
                gt_reject_count,
                gt_confidence,
                status.value,
                updated_at,
                template_family_id,
            ),
        )

    def link_fingerprint_to_family(self, hash_val: str, template_family_id: str):
        self._pg().execute(
            """
            UPDATE fingerprints
            SET template_family_id = ?
            WHERE fingerprint_hash = ?
            """,
            (template_family_id, hash_val),
        )

    def relink_fingerprints_to_family(
        self, fingerprint_hashes: List[str], template_family_id: str
    ) -> int:
        if not fingerprint_hashes:
            return 0
        placeholders = ",".join("?" * len(fingerprint_hashes))
        params: List[Any] = [template_family_id, *fingerprint_hashes]
        result = self._pg().execute(
            f"""
            UPDATE fingerprints
            SET template_family_id = ?
            WHERE fingerprint_hash IN ({placeholders})
            """,
            params,
        )
        try:
            return int(str(result).split()[-1])
        except Exception:
            return len(fingerprint_hashes)

    def get_family_representative_fingerprint(
        self, template_family_id: str
    ) -> Optional[FingerprintRecord]:
        row = self._pg().fetchone(
            """
            SELECT * FROM fingerprints
            WHERE template_family_id = ?
            ORDER BY confidence DESC, apply_count DESC, created_at ASC
            LIMIT 1
            """,
            (template_family_id,),
        )
        return self._row_to_fingerprint_record(row) if row else None

    def add_template_family_example(self, example: TemplateFamilyExample) -> int:
        row = self._pg().fetchone(
            """
            INSERT INTO template_family_examples (
                template_family_id, fingerprint_hash, invoice_path, example_metadata, created_at
            ) VALUES (?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                example.template_family_id,
                example.fingerprint_hash,
                example.invoice_path,
                json.dumps(example.example_metadata),
                example.created_at,
            ),
        )
        return int(row["id"]) if row else 0

    def get_template_family_examples(self, template_family_id: str) -> List[TemplateFamilyExample]:
        rows = self._pg().fetchall(
            """
            SELECT * FROM template_family_examples
            WHERE template_family_id = ?
            ORDER BY created_at, id
            """,
            (template_family_id,),
        )
        return [self._row_to_template_family_example(row) for row in rows]

    def add_template_family_version(self, version: TemplateFamilyVersion) -> int:
        row = self._pg().fetchone(
            """
            INSERT INTO template_family_versions (
                template_family_id, version, family_snapshot, change_reason, created_at
            ) VALUES (?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                version.template_family_id,
                version.version,
                json.dumps(version.family_snapshot),
                version.change_reason,
                version.created_at,
            ),
        )
        return int(row["id"]) if row else 0

    def get_template_family_versions(self, template_family_id: str) -> List[TemplateFamilyVersion]:
        rows = self._pg().fetchall(
            """
            SELECT * FROM template_family_versions
            WHERE template_family_id = ?
            ORDER BY version, id
            """,
            (template_family_id,),
        )
        return [self._row_to_template_family_version(row) for row in rows]

    def relink_template_family_examples(
        self,
        source_family_id: str,
        target_family_id: str,
        *,
        fingerprint_hashes: Optional[List[str]] = None,
        invoice_paths: Optional[List[str]] = None,
    ) -> int:
        fingerprint_hashes = fingerprint_hashes or []
        invoice_paths = invoice_paths or []
        clauses: List[str] = []
        params: List[Any] = [target_family_id, source_family_id]
        if fingerprint_hashes:
            placeholders = ",".join("?" * len(fingerprint_hashes))
            clauses.append(f"fingerprint_hash IN ({placeholders})")
            params.extend(fingerprint_hashes)
        if invoice_paths:
            placeholders = ",".join("?" * len(invoice_paths))
            clauses.append(f"invoice_path IN ({placeholders})")
            params.extend(invoice_paths)
        if not clauses:
            return 0
        result = self._pg().execute(
            f"""
            UPDATE template_family_examples
            SET template_family_id = ?
            WHERE template_family_id = ?
              AND ({" OR ".join(clauses)})
            """,
            params,
        )
        try:
            return int(str(result).split()[-1])
        except Exception:
            return 0

    def relink_processing_results_to_family(
        self,
        source_family_id: str,
        target_family_id: str,
        *,
        fingerprint_hashes: Optional[List[str]] = None,
        invoice_paths: Optional[List[str]] = None,
    ) -> int:
        fingerprint_hashes = fingerprint_hashes or []
        invoice_paths = invoice_paths or []
        if not fingerprint_hashes and not invoice_paths:
            return 0
        clauses: List[str] = []
        params: List[Any] = [source_family_id]
        if fingerprint_hashes:
            placeholders = ",".join("?" * len(fingerprint_hashes))
            clauses.append(f"fingerprint_hash IN ({placeholders})")
            params.extend(fingerprint_hashes)
        if invoice_paths:
            placeholders = ",".join("?" * len(invoice_paths))
            clauses.append(f"invoice_path IN ({placeholders})")
            params.extend(invoice_paths)
        rows = self._pg().fetchall(
            """
            SELECT id, provenance, diagnostics
            FROM processing_results
            WHERE template_family_id = ?
              AND (
            """
            + " OR ".join(clauses)
            + "\n              )",
            params,
        )
        updated = 0
        for row in rows:
            provenance_json = row.get("provenance")
            diagnostics_json = row.get("diagnostics")
            provenance = None
            diagnostics = None
            if provenance_json:
                provenance = Provenance.model_validate_json(provenance_json)
                provenance.template_family_id = target_family_id
            if diagnostics_json:
                diagnostics = ProcessingDiagnostics.model_validate_json(diagnostics_json)
                diagnostics.template_family_id = target_family_id
            self._pg().execute(
                """
                UPDATE processing_results
                SET template_family_id = ?, provenance = ?, diagnostics = ?
                WHERE id = ?
                """,
                (
                    target_family_id,
                    provenance.model_dump_json() if provenance else provenance_json,
                    diagnostics.model_dump_json() if diagnostics else diagnostics_json,
                    row["id"],
                ),
            )
            updated += 1
        return updated

    def retire_fingerprints(self, fingerprint_hashes: List[str]) -> int:
        if not fingerprint_hashes:
            return 0
        placeholders = ",".join("?" * len(fingerprint_hashes))
        result = self._pg().execute(
            f"""
            UPDATE fingerprints
            SET status = 'retired'
            WHERE fingerprint_hash IN ({placeholders})
            """,
            fingerprint_hashes,
        )
        try:
            return int(str(result).split()[-1])
        except Exception:
            return len(fingerprint_hashes)

    def update_fingerprint_confidence(
        self,
        hash_val: str,
        confidence: float,
        apply_count: int,
        reject_count: int,
        gt_apply_count: Optional[int],
        gt_reject_count: Optional[int],
        gt_confidence: Optional[float],
        status: TemplateStatus,
    ):
        params = (
            confidence,
            apply_count,
            reject_count,
            gt_apply_count,
            gt_reject_count,
            gt_confidence,
            status.value,
            hash_val,
        )
        self._pg().execute(
            """
            UPDATE fingerprints
            SET confidence = ?, apply_count = ?, reject_count = ?,
                gt_apply_count = COALESCE(?, gt_apply_count),
                gt_reject_count = COALESCE(?, gt_reject_count),
                gt_confidence = COALESCE(?, gt_confidence),
                status = ?
            WHERE fingerprint_hash = ?
            """,
            params,
        )

    def archive_fingerprint(
        self,
        hash_val: str,
        valid_from: str,
        valid_to: str,
        replaced_by: str,
        retirement_reason: str,
    ):
        row = self._pg().fetchone(
            "SELECT layout_template, confidence, status FROM fingerprints WHERE fingerprint_hash = ?",
            (hash_val,),
        )
        if row:
            self._pg().execute(
                """
                INSERT INTO fingerprint_history
                (fingerprint_hash, layout_template, confidence, status, valid_from, valid_to, replaced_by, retirement_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hash_val,
                    row["layout_template"],
                    row["confidence"],
                    row["status"],
                    valid_from,
                    valid_to,
                    replaced_by,
                    retirement_reason,
                ),
            )
            self._pg().execute("DELETE FROM fingerprints WHERE fingerprint_hash = ?", (hash_val,))

    def upsert_job_progress(self, job_id: str, current_item: str, status: str, started_at: str):
        params = (job_id, current_item, status, started_at)
        self._pg().execute(
            """
            INSERT INTO job_progress (job_id, current_item, status, started_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (job_id) DO UPDATE SET
                current_item = EXCLUDED.current_item,
                status = EXCLUDED.status
            """,
            params,
        )

    def _row_to_template_family_record(self, row: Mapping[str, Any]) -> TemplateFamilyRecord:
        row_dict = dict(row)
        from ...models import DocumentFamily, PageRole

        return TemplateFamilyRecord(
            template_family_id=row_dict["template_family_id"],
            provider_name=row_dict.get("provider_name"),
            country_code=row_dict.get("country_code"),
            document_family=DocumentFamily(row_dict["document_family"]),
            stable_anchor_regions=json.loads(row_dict["stable_anchor_regions"])
            if row_dict.get("stable_anchor_regions")
            else {},
            anchor_summary=json.loads(row_dict["anchor_summary"])
            if row_dict.get("anchor_summary")
            else {},
            page_role_expectations=[
                PageRole(role) for role in json.loads(row_dict["page_role_expectations"])
            ]
            if row_dict.get("page_role_expectations")
            else [],
            summary_area_anchors=json.loads(row_dict["summary_area_anchors"])
            if row_dict.get("summary_area_anchors")
            else {},
            variable_region_masks=json.loads(row_dict["variable_region_masks"])
            if row_dict.get("variable_region_masks")
            else [],
            extraction_profile=json.loads(row_dict["extraction_profile"])
            if row_dict.get("extraction_profile")
            else {},
            confidence=row_dict["confidence"],
            apply_count=int(row_dict.get("apply_count") or 0),
            reject_count=int(row_dict.get("reject_count") or 0),
            gt_apply_count=int(row_dict.get("gt_apply_count") or 0),
            gt_reject_count=int(row_dict.get("gt_reject_count") or 0),
            gt_confidence=float(row_dict.get("gt_confidence") or 0.0),
            status=TemplateStatus(row_dict["status"]),
            created_at=row_dict["created_at"],
            updated_at=row_dict.get("updated_at"),
        )

    def _row_to_fingerprint_record(self, row: Mapping[str, Any]) -> FingerprintRecord:
        from ...models import PageFingerprint

        row_dict = dict(row)
        return FingerprintRecord(
            hash=row_dict["fingerprint_hash"],
            layout_template=json.loads(row_dict["layout_template"]),
            template_family_id=row_dict.get("template_family_id"),
            provider_name=row_dict.get("provider_name"),
            country_code=row_dict.get("country_code"),
            page_fingerprints=[
                PageFingerprint(**p) for p in json.loads(row_dict["page_fingerprints"])
            ],
            confidence=row_dict["confidence"],
            apply_count=row_dict["apply_count"],
            reject_count=row_dict["reject_count"],
            gt_apply_count=int(row_dict.get("gt_apply_count") or 0),
            gt_reject_count=int(row_dict.get("gt_reject_count") or 0),
            gt_confidence=float(row_dict.get("gt_confidence") or 0.0),
            status=TemplateStatus(row_dict["status"]),
            version=row_dict["version"],
            created_at=row_dict["created_at"],
            last_used=row_dict.get("last_used"),
        )

    def _row_to_template_family_example(self, row: Mapping[str, Any]) -> TemplateFamilyExample:
        row_dict = dict(row)
        return TemplateFamilyExample(
            id=row_dict.get("id"),
            template_family_id=row_dict["template_family_id"],
            fingerprint_hash=row_dict.get("fingerprint_hash"),
            invoice_path=row_dict.get("invoice_path"),
            example_metadata=json.loads(row_dict["example_metadata"])
            if row_dict.get("example_metadata")
            else {},
            created_at=row_dict["created_at"],
        )

    def _row_to_template_family_version(self, row: Mapping[str, Any]) -> TemplateFamilyVersion:
        row_dict = dict(row)
        return TemplateFamilyVersion(
            id=row_dict.get("id"),
            template_family_id=row_dict["template_family_id"],
            version=row_dict["version"],
            family_snapshot=json.loads(row_dict["family_snapshot"])
            if row_dict.get("family_snapshot")
            else {},
            change_reason=row_dict.get("change_reason"),
            created_at=row_dict["created_at"],
        )
