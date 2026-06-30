from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


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
    parser.add_argument("--max-symbols", type=int, default=800)
    parser.add_argument("--history-years", type=int, default=4)
    parser.add_argument("--skip-finance", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    config = StrategyConfig(top_n=20)
    request = DataRequest(
        max_symbols=args.max_symbols,
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
    output = write_snapshot(result, args.output, config)
    print(f"Snapshot written to {output}")

    if args.publish:
        repo_id = os.getenv("HF_SNAPSHOT_REPO_ID", "").strip()
        token = os.getenv("HF_TOKEN", "").strip()
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


if __name__ == "__main__":
    main()
