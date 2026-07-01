from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from a_share_recommender.config import StrategyConfig
from a_share_recommender.data_providers import DataRequest
from a_share_recommender.pipeline import run_pipeline
from a_share_recommender.snapshot import write_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and optionally publish the public A-share snapshot.")
    parser.add_argument("--output", default="data/snapshot")
    parser.add_argument("--max-symbols", type=int, default=None, help="Optional bounded test run; omit for all listed A-shares.")
    parser.add_argument("--history-years", type=int, default=4)
    parser.add_argument("--skip-finance", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--bootstrap-previous", action="store_true")
    args = parser.parse_args()

    repo_id = os.getenv("HF_SNAPSHOT_REPO_ID", "").strip()
    token = os.getenv("HF_TOKEN", "").strip()
    if args.publish or args.bootstrap_previous:
        _bootstrap_previous_market(repo_id, token, ROOT / "data" / "cache")

    config = StrategyConfig(top_n=20)
    request = DataRequest(
        max_symbols=args.max_symbols,
        all_listed=args.max_symbols is None,
        history_years=args.history_years,
        use_finance=not args.skip_finance,
        force_refresh=args.force_refresh,
        allow_sample_fallback=False,
        full_market_scan=True,
    )
    result = run_pipeline(
        config,
        prefer_tushare=os.getenv("PREFER_TUSHARE", "").lower() in {"1", "true", "yes"},
        tushare_token=os.getenv("TUSHARE_TOKEN") or None,
        data_request=request,
    )
    expected_symbols = result.provider_status.requested_symbols or args.max_symbols
    if expected_symbols is None:
        raise RuntimeError("数据源没有报告全市场目标股票数，拒绝发布快照。")
    universe_path = ROOT / "data" / "cache" / "akshare_universe.parquet"
    listed_universe = pd.read_parquet(universe_path) if universe_path.exists() else None
    output = write_snapshot(
        result,
        args.output,
        config,
        expected_symbols=expected_symbols,
        listed_universe=listed_universe,
    )
    print(f"Snapshot written to {output}")

    if args.publish:
        if not repo_id or not token:
            raise RuntimeError("HF_SNAPSHOT_REPO_ID and HF_TOKEN are required for --publish")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(output),
            path_in_repo=".",
            commit_message="Publish latest A-share snapshot",
        )
        print(f"Snapshot published to dataset {repo_id}")


def _bootstrap_previous_market(repo_id: str, token: str, cache_dir: Path) -> Path | None:
    if not repo_id:
        return None
    try:
        from huggingface_hub import hf_hub_download

        manifest_path = Path(
            hf_hub_download(repo_id, "manifest.json", repo_type="dataset", token=token or None)
        )
    except Exception as exc:
        print(f"Previous snapshot bootstrap unavailable: {type(exc).__name__}")
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = None
    expected_hash = manifest.get("builder_files", {}).get("provider_market.parquet")
    if expected_hash:
        source = Path(
            hf_hub_download(repo_id, "provider_market.parquet", repo_type="dataset", token=token or None)
        )
        if _sha256(source) != expected_hash:
            raise RuntimeError("Previous full-market fallback failed SHA-256 validation")
        destination = cache_dir / "akshare_previous_snapshot.parquet"
        shutil.copyfile(source, destination)
        print(f"Previous full-market fallback copied to {destination}")
    else:
        print("Previous snapshot has no full-market fallback; continuing without market bootstrap.")

    try:
        universe_source = Path(
            hf_hub_download(repo_id, "listed_universe.parquet", repo_type="dataset", token=token or None)
        )
        universe_hash = manifest.get("builder_files", {}).get("listed_universe.parquet")
        if universe_hash and _sha256(universe_source) != universe_hash:
            raise RuntimeError("Previous listed-universe fallback failed SHA-256 validation")
        universe = pd.read_parquet(universe_source)
        required = {"code", "name", "board"}
        if not required.issubset(universe.columns) or universe["code"].nunique() < 1000:
            raise RuntimeError("Previous listed-universe fallback is incomplete")
        universe_destination = cache_dir / "akshare_universe.parquet"
        shutil.copyfile(universe_source, universe_destination)
        print(f"Previous listed universe copied to {universe_destination}")
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"Previous listed-universe bootstrap unavailable: {type(exc).__name__}")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
