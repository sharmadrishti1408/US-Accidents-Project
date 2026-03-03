#!/usr/bin/env python3
# =============================================================================
# run_pipeline.py
# US Accidents Big Data ML Assignment - Full Pipeline Orchestrator
# Runs all 4 notebook stages programmatically via nbconvert
# Usage: python scripts/run_pipeline.py [--stage 1|2|3|4] [--all]
# =============================================================================

import os
import sys
import time
import argparse
import logging
import subprocess
import json
from pathlib import Path

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/pipeline_run.log")
    ]
)
logger = logging.getLogger("PipelineRunner")

# ---- Project paths ----
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DATA_DIR      = PROJECT_ROOT / "data" / "samples"
TABLEAU_DIR   = PROJECT_ROOT / "tableau"

# ---- Notebook execution order ----
PIPELINE_STAGES = {
    1: "1_data_ingestion.ipynb",
    2: "2_feature_engineering.ipynb",
    3: "3_model_training.ipynb",
    4: "4_evaluation.ipynb",
}


def check_prerequisites():
    """Validate that required files and directories exist before running."""
    raw_csv = Path("Drishti/US_Accidents_March23.csv")
    if not raw_csv.exists():
        logger.error(f"Raw dataset not found: {raw_csv}")
        sys.exit(1)

    # Check dataset size (~2.9GB expected)
    size_gb = raw_csv.stat().st_size / (1024 ** 3)
    logger.info(f"Dataset found: {raw_csv} ({size_gb:.2f} GB)")

    if size_gb < 2.0:
        logger.warning(f"Dataset smaller than expected ({size_gb:.2f}GB). Proceed with caution.")

    # Check PySpark availability
    try:
        import pyspark
        logger.info(f"PySpark {pyspark.__version__} available.")
    except ImportError:
        logger.error("PySpark not found. Run setup_environment.sh first.")
        sys.exit(1)

    # Ensure necessary directories exist
    for d in [DATA_DIR, DATA_DIR / "splits", DATA_DIR / "models", TABLEAU_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("All prerequisites satisfied.")


def run_notebook(notebook_name, stage_num, timeout_seconds=7200):
    """Execute a Jupyter notebook using nbconvert and return success/failure."""
    nb_path = NOTEBOOKS_DIR / notebook_name
    output_path = NOTEBOOKS_DIR / notebook_name.replace(".ipynb", "_executed.ipynb")

    if not nb_path.exists():
        logger.error(f"Notebook not found: {nb_path}")
        return False, 0.0

    logger.info(f"=" * 60)
    logger.info(f"STAGE {stage_num}: {notebook_name}")
    logger.info(f"=" * 60)

    start_time = time.time()

    # Run notebook via nbconvert (executes all cells in order)
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout_seconds}",
        "--ExecutePreprocessor.allow_errors=False",
        "--output", str(output_path),
        str(nb_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 60  # Grace period beyond notebook timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"Stage {stage_num} completed in {elapsed:.1f}s")
            return True, elapsed
        else:
            logger.error(f"Stage {stage_num} FAILED after {elapsed:.1f}s")
            logger.error(f"STDOUT: {result.stdout[-2000:]}")
            logger.error(f"STDERR: {result.stderr[-2000:]}")
            return False, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(f"Stage {stage_num} TIMED OUT after {elapsed:.1f}s (limit: {timeout_seconds}s)")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Stage {stage_num} ERROR: {e}")
        return False, elapsed


def run_pipeline(stages=None, fail_fast=True):
    """Run specified pipeline stages (default: all stages in order)."""
    if stages is None:
        stages = list(PIPELINE_STAGES.keys())

    logger.info("Starting US Accidents ML Pipeline")
    logger.info(f"Stages to run: {stages}")

    check_prerequisites()

    pipeline_results = {}
    total_start = time.time()

    for stage_num in stages:
        if stage_num not in PIPELINE_STAGES:
            logger.warning(f"Unknown stage: {stage_num}. Skipping.")
            continue

        nb_name = PIPELINE_STAGES[stage_num]
        success, elapsed = run_notebook(nb_name, stage_num)

        pipeline_results[stage_num] = {
            "notebook": nb_name,
            "success": success,
            "elapsed_s": round(elapsed, 1)
        }

        if not success and fail_fast:
            logger.error(f"Pipeline stopped at stage {stage_num} due to failure.")
            break

    total_elapsed = time.time() - total_start

    # ---- Print pipeline summary ----
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    for stage_num, res in pipeline_results.items():
        status = "SUCCESS" if res["success"] else "FAILED"
        logger.info(f"  Stage {stage_num}: {res['notebook']:<35} [{status}] {res['elapsed_s']:.1f}s")

    logger.info(f"\nTotal pipeline time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # ---- Save pipeline run report ----
    report = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": pipeline_results,
        "total_elapsed_s": round(total_elapsed, 1),
        "overall_success": all(r["success"] for r in pipeline_results.values())
    }

    report_path = DATA_DIR / "pipeline_run_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Pipeline report saved: {report_path}")

    return report["overall_success"]


def main():
    parser = argparse.ArgumentParser(description="US Accidents ML Pipeline Runner")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4],
                        help="Run a specific stage (1-4)")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages sequentially (default behavior)")
    parser.add_argument("--no-fail-fast", action="store_true",
                        help="Continue pipeline even if a stage fails")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check prerequisites without running")

    args = parser.parse_args()

    if args.check_only:
        check_prerequisites()
        return

    fail_fast = not args.no_fail_fast

    if args.stage:
        stages = [args.stage]
    else:
        stages = list(PIPELINE_STAGES.keys())

    success = run_pipeline(stages, fail_fast=fail_fast)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
