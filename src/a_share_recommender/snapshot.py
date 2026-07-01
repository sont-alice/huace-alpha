from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd

from .config import StrategyConfig
from .data_providers import ProviderStatus
from .modeling import ModelResult
from .pipeline import PipelineResult


SCHEMA_VERSION = 2
APP_SNAPSHOT_FILES = (
    "result.json",
    "recommendations.parquet",
    "latest_scored.parquet",
    "equity_curve.parquet",
    "market_history.parquet",
)
BUILDER_SNAPSHOT_FILES = ("provider_market.parquet",)
SNAPSHOT_FILES = APP_SNAPSHOT_FILES


def public_snapshot_mode() -> bool:
    return os.getenv("PUBLIC_SNAPSHOT_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def write_snapshot(
    result: PipelineResult,
    destination: Path | str,
    config: StrategyConfig,
    expected_symbols: int | None = None,
) -> Path:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        recommendations = result.recommendations.sort_values("composite_score", ascending=False).reset_index(drop=True)
        market_symbol_count = int(result.market["code"].nunique())
        scored_symbol_count = int(result.latest_scored["code"].nunique())
        if expected_symbols is not None and (
            market_symbol_count != expected_symbols or scored_symbol_count != expected_symbols
        ):
            raise RuntimeError(
                f"快照覆盖未达标：要求 {expected_symbols} 只，"
                f"行情 {market_symbol_count} 只，评分 {scored_symbol_count} 只。"
            )
        if not recommendations["composite_score"].is_monotonic_decreasing:
            raise RuntimeError("推荐结果未按综合评分降序排列。")
        recommendations.to_parquet(temp_dir / "recommendations.parquet", index=False)
        result.latest_scored.to_parquet(temp_dir / "latest_scored.parquet", index=False)
        result.equity_curve.to_parquet(temp_dir / "equity_curve.parquet", index=False)
        result.market[["date", "code", "close"]].to_parquet(temp_dir / "market_history.parquet", index=False)
        result.market.to_parquet(temp_dir / "provider_market.parquet", index=False)

        payload = {
            "provider_status": asdict(result.provider_status),
            "model_result": {
                "train_rows": result.model_result.train_rows,
                "test_rows": result.model_result.test_rows,
                "train_end": result.model_result.train_end.isoformat(),
                "test_start": result.model_result.test_start.isoformat(),
            },
            "metrics": result.metrics,
            "gate_ok": result.gate_ok,
            "gate_reasons": result.gate_reasons,
            "data_date": result.data_date.isoformat(),
            "availability": result.availability,
            "strategy_config": asdict(config),
        }
        _write_json(temp_dir / "result.json", payload)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_date": result.data_date.isoformat(),
            "requested_symbol_count": expected_symbols or scored_symbol_count,
            "market_symbol_count": market_symbol_count,
            "scored_symbol_count": scored_symbol_count,
            "symbol_count": scored_symbol_count,
            "row_count": int(len(result.market)),
            "provider_mode": result.provider_status.mode,
            "requested_symbols": result.provider_status.requested_symbols,
            "refreshed_symbols": result.provider_status.refreshed_symbols,
            "baseline_filled_symbols": result.provider_status.baseline_filled_symbols,
            "board_counts": {
                str(board): int(count)
                for board, count in result.latest_scored["board"].value_counts().items()
            },
            "app_files": {name: _sha256(temp_dir / name) for name in APP_SNAPSHOT_FILES},
            "builder_files": {name: _sha256(temp_dir / name) for name in BUILDER_SNAPSHOT_FILES},
        }
        _write_json(temp_dir / "manifest.json", manifest)

        if destination.exists():
            shutil.rmtree(destination)
        temp_dir.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def load_snapshot(source: Path | str) -> PipelineResult:
    source = Path(source)
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"线上快照不存在：{manifest_path}")

    manifest = _read_json(manifest_path)
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, SCHEMA_VERSION}:
        raise RuntimeError("线上快照版本不兼容，请重新生成。")
    file_hashes = manifest.get("files", {}) if schema_version == 1 else manifest.get("app_files", {})
    if set(file_hashes) != set(APP_SNAPSHOT_FILES):
        raise RuntimeError("线上快照文件清单不完整。")
    for name, expected_hash in file_hashes.items():
        path = source / name
        if not path.exists() or _sha256(path) != expected_hash:
            raise RuntimeError(f"线上快照文件校验失败：{name}")

    payload = _read_json(source / "result.json")
    model_payload = payload["model_result"]
    original_status = payload["provider_status"]
    status = ProviderStatus(
        mode=f"snapshot-{original_status['mode']}",
        message=f"共享预计算快照；原数据状态：{original_status['message']}",
        rows=int(original_status["rows"]),
        requested_symbols=int(original_status.get("requested_symbols", manifest.get("requested_symbol_count", 0))),
        refreshed_symbols=int(original_status.get("refreshed_symbols", 0)),
        baseline_filled_symbols=int(original_status.get("baseline_filled_symbols", 0)),
    )
    model_result = ModelResult(
        model=None,
        train_rows=int(model_payload["train_rows"]),
        test_rows=int(model_payload["test_rows"]),
        train_end=pd.Timestamp(model_payload["train_end"]),
        test_start=pd.Timestamp(model_payload["test_start"]),
    )
    return PipelineResult(
        provider_status=status,
        model_result=model_result,
        metrics={key: float(value) for key, value in payload["metrics"].items()},
        gate_ok=bool(payload["gate_ok"]),
        gate_reasons=list(payload["gate_reasons"]),
        equity_curve=pd.read_parquet(source / "equity_curve.parquet"),
        recommendations=pd.read_parquet(source / "recommendations.parquet"),
        market=pd.read_parquet(source / "market_history.parquet"),
        latest_scored=pd.read_parquet(source / "latest_scored.parquet"),
        data_date=pd.Timestamp(payload["data_date"]),
        availability=dict(payload["availability"]),
    )


def load_configured_snapshot() -> PipelineResult:
    local_dir = Path(os.getenv("SNAPSHOT_DIR", "data/snapshot"))
    if (local_dir / "manifest.json").exists():
        return load_snapshot(local_dir)

    repo_id = os.getenv("HF_SNAPSHOT_REPO_ID", "").strip()
    if not repo_id:
        raise RuntimeError("尚未配置线上数据快照。")
    try:
        from huggingface_hub import snapshot_download

        downloaded = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=os.getenv("HF_SNAPSHOT_REVISION", "main"),
            allow_patterns=["manifest.json", *APP_SNAPSHOT_FILES],
            token=os.getenv("HF_TOKEN") or None,
        )
    except Exception as exc:
        raise RuntimeError(f"下载线上数据快照失败：{type(exc).__name__}") from exc
    return load_snapshot(downloaded)


def read_manifest(source: Path | str) -> dict[str, Any]:
    return _read_json(Path(source) / "manifest.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
