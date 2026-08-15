"""
Trial Ledger -- registro append-only de cada configuracion probada.

Cada vez que cualquier agente entrena o evalua una configuracion
(incluyendo baselines y variaciones descartadas), se registra una fila.
Sin esto, V es una suposicion, no un dato.

Usa SQLite (un archivo, sin servidor).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------

@dataclass
class TrialRecord:
    """Un trial = una configuracion evaluada, exitosa o no."""

    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent: str = ""                  # que agente genero el trial
    model_name: str = ""             # nombre legible del modelo
    config_hash: str = ""            # hash de hiperparametros -> detecta duplicados
    hyperparams: str = ""            # JSON serializado de los hiperparametros
    sharpe_is: float = 0.0           # Sharpe in-sample
    sharpe_oos: float = 0.0          # Sharpe out-of-sample (CPCV)
    returns_path: str = ""           # ruta al archivo de retornos OOS
    n_observations: int = 0          # T de este trial
    d_used: Optional[float] = None    # orden FFD usado en las features
    d_selection_method: str = ""      # rule_based_adf | performance_based
    annualization_factor: float = 0.0
    frequency: str = ""
    config_version: str = ""
    git_commit: str = ""             # hash del commit activo
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @staticmethod
    def hash_config(config: Dict[str, Any]) -> str:
        """Hash determinista de una configuracion para detectar duplicados."""
        serialized = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _get_git_commit() -> str:
    """Intenta obtener el hash del commit activo de git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Base de datos
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id       TEXT PRIMARY KEY,
    agent          TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    config_hash    TEXT NOT NULL,
    hyperparams    TEXT,
    sharpe_is      REAL,
    sharpe_oos     REAL,
    returns_path   TEXT,
    n_observations INTEGER,
    d_used         REAL,
    d_selection_method TEXT,
    annualization_factor REAL,
    frequency      TEXT,
    config_version TEXT,
    git_commit     TEXT,
    timestamp      TEXT NOT NULL
);
"""

_INSERT = """
INSERT INTO trials (
    trial_id, agent, model_name, config_hash, hyperparams,
    sharpe_is, sharpe_oos, returns_path, n_observations,
    d_used, d_selection_method, annualization_factor, frequency,
    config_version, git_commit, timestamp
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_MIGRATED_COLUMNS = {
    "d_used": "REAL",
    "d_selection_method": "TEXT",
    "annualization_factor": "REAL",
    "frequency": "TEXT",
    "config_version": "TEXT",
}


def _connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_CREATE_TABLE)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
    for column, sql_type in _MIGRATED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE trials ADD COLUMN {column} {sql_type}")
    conn.commit()
    return conn


def log_trial(
    record: TrialRecord,
    ledger_path: str | Path = "config/trial_ledger.db",
) -> None:
    """Registra un trial en el ledger (append-only)."""
    if not record.git_commit or record.git_commit == "":
        record.git_commit = _get_git_commit()

    conn = _connect(ledger_path)
    try:
        conn.execute(_INSERT, (
            record.trial_id,
            record.agent,
            record.model_name,
            record.config_hash,
            record.hyperparams,
            record.sharpe_is,
            record.sharpe_oos,
            record.returns_path,
            record.n_observations,
            record.d_used,
            record.d_selection_method,
            record.annualization_factor,
            record.frequency,
            record.config_version,
            record.git_commit,
            record.timestamp,
        ))
        conn.commit()
        log.info(
            "trial_logged",
            trial_id=record.trial_id,
            agent=record.agent,
            model_name=record.model_name,
            sharpe_oos=record.sharpe_oos,
        )
    finally:
        conn.close()


def count_trials(
    ledger_path: str | Path = "config/trial_ledger.db",
    cutoff_timestamp: Optional[str] = None,
) -> int:
    """Devuelve V real -- el numero de trials registrados.

    Si se pasa cutoff_timestamp (ISO format), solo cuenta trials
    registrados hasta ese momento. Esto es critico: el DSR de un
    modelo temprano NO debe recalcularse con el V final.
    """
    conn = _connect(ledger_path)
    try:
        if cutoff_timestamp is not None:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM trials WHERE timestamp <= ?",
                (cutoff_timestamp,),
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM trials")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def get_all_trials(
    ledger_path: str | Path = "config/trial_ledger.db",
    cutoff_timestamp: Optional[str] = None,
) -> List[TrialRecord]:
    """Recupera todos los trials (opcionalmente hasta un cutoff)."""
    conn = _connect(ledger_path)
    try:
        if cutoff_timestamp is not None:
            cursor = conn.execute(
                """SELECT trial_id, agent, model_name, config_hash,
                          hyperparams, sharpe_is, sharpe_oos, returns_path,
                          n_observations, d_used, d_selection_method,
                          annualization_factor, frequency, config_version,
                          git_commit, timestamp
                   FROM trials WHERE timestamp <= ? ORDER BY timestamp""",
                (cutoff_timestamp,),
            )
        else:
            cursor = conn.execute(
                """SELECT trial_id, agent, model_name, config_hash,
                          hyperparams, sharpe_is, sharpe_oos, returns_path,
                          n_observations, d_used, d_selection_method,
                          annualization_factor, frequency, config_version,
                          git_commit, timestamp
                   FROM trials ORDER BY timestamp"""
            )
        rows = cursor.fetchall()
        return [
            TrialRecord(
                trial_id=r[0], agent=r[1], model_name=r[2],
                config_hash=r[3], hyperparams=r[4],
                sharpe_is=r[5], sharpe_oos=r[6],
                returns_path=r[7], n_observations=r[8],
                d_used=r[9], d_selection_method=r[10],
                annualization_factor=r[11], frequency=r[12],
                config_version=r[13], git_commit=r[14], timestamp=r[15],
            )
            for r in rows
        ]
    finally:
        conn.close()


def get_sharpe_oos_series(
    ledger_path: str | Path = "config/trial_ledger.db",
    cutoff_timestamp: Optional[str] = None,
) -> List[float]:
    """Devuelve la serie de Sharpe OOS de todos los trials.
    Util para calcular la correlacion promedio entre trials."""
    trials = get_all_trials(ledger_path, cutoff_timestamp)
    return [t.sharpe_oos for t in trials]
