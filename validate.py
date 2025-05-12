import os
import logging
from datetime import datetime
from ultralytics import YOLO

# ========== Configuration ==========
RUNS_DIR = "runs"
DATA_YAML = "./datasets/data.yaml"
LOG_FILE = "replot_missing_results.log"
# ===================================

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info("===== Starting re-plot script =====")

for run_name in os.listdir(RUNS_DIR):
    run_path = os.path.join(RUNS_DIR, run_name)
    weights_path = os.path.join(run_path, "weights", "best.pt")
    results_img_path = os.path.join(run_path, "results.png")

    if not os.path.isdir(run_path):
        continue

    logging.info(f"Checking run: {run_name}")

    if os.path.exists(results_img_path):
        logging.info("✅ results.png already exists. Skipping.")
        continue

    if os.path.exists(weights_path):
        logging.info("📊 results.png missing. Validating model to regenerate...")
        try:
            model = YOLO(weights_path)
            model.val(data=DATA_YAML, plots=True)
            logging.info("✅ Successfully regenerated results.png")
        except Exception as e:
            logging.error(f"❌ Validation failed for {run_name}: {e}")
    else:
        logging.warning("⚠️ best.pt not found. Skipping.")

logging.info("===== Script completed =====\n")
