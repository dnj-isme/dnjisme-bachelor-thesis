from ultralytics import YOLO
import os
import datetime
import csv
import yaml # Untuk membaca data.yaml

# --- Konfigurasi ---
STATUS_LOG = "validation_status.log"
ERROR_LOG = "validation_error.log"

DATA_YAML_PATH = "./datasets/data.yaml"
TRAINED_MODELS_BASE_DIR = "./runs"
EVAL_PROJECT_DIR = "./eval_runs"
RESULTS_CSV_FILE = os.path.join(EVAL_PROJECT_DIR, "all_models_validation_metrics.csv")

IMG_SIZE = 640
BATCH_SIZE_VAL = 16
DEVICE_VAL = None 
# --------------------

def log_message(message_type, description, log_file=STATUS_LOG):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] - {message_type} - {description}\n"
    with open(log_file, "a") as lf:
        lf.write(log_entry)

def find_best_model_paths(base_dir):
    models_to_evaluate = []
    log_message("Info", f"Scanning directory {base_dir} for trained models (best.pt)...")

    if not os.path.isdir(base_dir):
        log_message("Error", f"Base directory for trained models not found: {base_dir}", ERROR_LOG)
        print(f"❌ Base directory for trained models not found: {base_dir}")
        return models_to_evaluate

    for folder_name in os.listdir(base_dir):
        potential_run_dir = os.path.join(base_dir, folder_name)
        if os.path.isdir(potential_run_dir):
            best_model_path = os.path.join(potential_run_dir, "weights", "best.pt")
            if os.path.exists(best_model_path):
                models_to_evaluate.append((folder_name, best_model_path))
                log_message("Info", f"Found model {best_model_path} for evaluation (label: {folder_name}).")

    if not models_to_evaluate:
        log_message("Warning", f"No 'best.pt' models found in {base_dir}.")
        print(f"⚠️ No 'best.pt' models found in {base_dir}.")
        
    return models_to_evaluate

if __name__ == "__main__":
    log_message("Status", "Validation process started.")
    log_message("Configuration", f"Data YAML: {DATA_YAML_PATH}, Trained Models Dir: {TRAINED_MODELS_BASE_DIR}, Eval Project: {EVAL_PROJECT_DIR}")

    if not os.path.exists(DATA_YAML_PATH):
        log_message("Error", f"Data YAML file not found: {DATA_YAML_PATH}", ERROR_LOG)
        print(f"❌ Data YAML file not found: {DATA_YAML_PATH}")
        exit(1)

    os.makedirs(EVAL_PROJECT_DIR, exist_ok=True)

    class_names = []
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get('names', [])
        if not class_names:
            log_message("Warning", "Could not read class names from data.yaml. Per-class metrics might be limited or have generic names.")
            print("⚠️ Could not read class names from data.yaml. CSV header for per-class metrics might be limited.")
            # Fallback jika class_names kosong, tapi nc (number of classes) mungkin diperlukan dari data_config
            num_classes_from_config = data_config.get('nc', 0)
            if num_classes_from_config > 0 and not class_names:
                class_names = [f"Class{i}" for i in range(num_classes_from_config)]

    except Exception as e:
        log_message("Error", f"Failed to read class names from {DATA_YAML_PATH}: {e}", ERROR_LOG)
        print(f"❌ Failed to read class names from {DATA_YAML_PATH}: {e}")
        # Jika tidak bisa baca nama kelas, kita tidak bisa membuat header per kelas yang benar
        # Skrip bisa tetap jalan tapi kolom per kelas mungkin tidak ada atau salah


    models_to_evaluate = find_best_model_paths(TRAINED_MODELS_BASE_DIR)

    if not models_to_evaluate:
        log_message("Status", "No models to evaluate. Exiting.")
        print("No models found to evaluate. Exiting.")
        exit(0)

    print(f"\n--- Starting Evaluation on Test Set for {len(models_to_evaluate)} model(s) ---")

    all_metrics_data_list = [] 

    for model_label, model_path in models_to_evaluate:
        print(f"\nEvaluating model: {model_label} from {model_path}")
        log_message("Evaluation", f"Starting evaluation for model {model_label} using {model_path}")

        current_model_metrics_dict = {"Model Label": model_label}

        try:
            model = YOLO(model_path)
            eval_run_name = model_label 
            
            metrics_obj = model.val(
                data=DATA_YAML_PATH,
                split='test',
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE_VAL,
                project=EVAL_PROJECT_DIR,
                name=eval_run_name,
                device=DEVICE_VAL,
                exist_ok=True
            )
            
            log_message("Evaluation", f"Finished evaluation for model {model_label}")

            # Ekstrak metrik utama (keseluruhan)
            if hasattr(metrics_obj, 'box') and metrics_obj.box:
                current_model_metrics_dict["mAP50 (Overall)"] = f"{metrics_obj.box.map50:.4f}"
                current_model_metrics_dict["mAP50-95 (Overall)"] = f"{metrics_obj.box.map:.4f}"
                
                if hasattr(metrics_obj, 'results_dict') and metrics_obj.results_dict:
                    p_overall = metrics_obj.results_dict.get('metrics/precision(B)', 0.0)
                    r_overall = metrics_obj.results_dict.get('metrics/recall(B)', 0.0)
                    current_model_metrics_dict["Precision (Overall)"] = f"{p_overall:.4f}"
                    current_model_metrics_dict["Recall (Overall)"] = f"{r_overall:.4f}"
                    if p_overall + r_overall > 0:
                        f1_overall = 2 * (p_overall * r_overall) / (p_overall + r_overall)
                        current_model_metrics_dict["F1-Score (Overall)"] = f"{f1_overall:.4f}"
                    else:
                        current_model_metrics_dict["F1-Score (Overall)"] = "0.0000"
                elif hasattr(metrics_obj.box, 'mean_results') and len(metrics_obj.box.mean_results) >= 3:
                    current_model_metrics_dict["Precision (Overall)"] = f"{metrics_obj.box.mean_results[0]:.4f}"
                    current_model_metrics_dict["Recall (Overall)"] = f"{metrics_obj.box.mean_results[1]:.4f}"
                    current_model_metrics_dict["F1-Score (Overall)"] = f"{metrics_obj.box.mean_results[2]:.4f}"
                else:
                    current_model_metrics_dict["Precision (Overall)"] = "N/A"
                    current_model_metrics_dict["Recall (Overall)"] = "N/A"
                    current_model_metrics_dict["F1-Score (Overall)"] = "N/A"
            else:
                current_model_metrics_dict["mAP50 (Overall)"] = "N/A"; current_model_metrics_dict["mAP50-95 (Overall)"] = "N/A"
                current_model_metrics_dict["Precision (Overall)"] = "N/A"; current_model_metrics_dict["Recall (Overall)"] = "N/A"; current_model_metrics_dict["F1-Score (Overall)"] = "N/A"

            if hasattr(metrics_obj, 'speed') and metrics_obj.speed:
                current_model_metrics_dict["Inference Speed (ms/img)"] = f"{metrics_obj.speed.get('inference', 0.0):.2f}"
                current_model_metrics_dict["Preprocessing Speed (ms/img)"] = f"{metrics_obj.speed.get('preprocess', 0.0):.2f}"
                current_model_metrics_dict["NMS Speed (ms/img)"] = f"{metrics_obj.speed.get('postprocess', 0.0):.2f}"
            else:
                current_model_metrics_dict["Inference Speed (ms/img)"] = "N/A"; current_model_metrics_dict["Preprocessing Speed (ms/img)"] = "N/A"; current_model_metrics_dict["NMS Speed (ms/img)"] = "N/A"

            # Ekstrak metrik per kelas
            if class_names and hasattr(metrics_obj.box, 'ap_class_index') and \
                hasattr(metrics_obj.box, 'p') and hasattr(metrics_obj.box, 'r') and \
                hasattr(metrics_obj.box, 'f1') and hasattr(metrics_obj.box, 'ap50'):

                per_class_p_map = {}
                per_class_r_map = {}
                per_class_f1_map = {}
                per_class_ap50_map = {}

                # Pastikan panjang array metrik konsisten dengan ap_class_index
                if len(metrics_obj.box.ap_class_index) == len(metrics_obj.box.p) == \
                    len(metrics_obj.box.r) == len(metrics_obj.box.f1) == len(metrics_obj.box.ap50):
                    for i, original_idx in enumerate(metrics_obj.box.ap_class_index):
                        per_class_p_map[original_idx] = metrics_obj.box.p[i]
                        per_class_r_map[original_idx] = metrics_obj.box.r[i]
                        per_class_f1_map[original_idx] = metrics_obj.box.f1[i]
                        per_class_ap50_map[original_idx] = metrics_obj.box.ap50[i] # ap50 adalah AP@0.5 untuk kelas ini
                else:
                    log_message("Warning", f"Mismatch in lengths of per-class metric arrays for {model_label}. Some per-class metrics might be 0 or N/A.")

                for original_class_idx, class_name_actual in enumerate(class_names):
                    current_model_metrics_dict[f"Precision ({class_name_actual})"] = f"{per_class_p_map.get(original_class_idx, 0.0):.4f}"
                    current_model_metrics_dict[f"Recall ({class_name_actual})"] = f"{per_class_r_map.get(original_class_idx, 0.0):.4f}"
                    current_model_metrics_dict[f"F1-Score ({class_name_actual})"] = f"{per_class_f1_map.get(original_class_idx, 0.0):.4f}"
                    current_model_metrics_dict[f"AP@0.5 ({class_name_actual})"] = f"{per_class_ap50_map.get(original_class_idx, 0.0):.4f}"
            
            else: # Jika tidak bisa mendapatkan metrik per kelas
                log_message("Warning", f"Could not retrieve one or more per-class metric arrays for {model_label}. Setting per-class metrics to N/A.")
                if class_names: # Hanya jika class_names berhasil dibaca
                    for cn in class_names:
                        current_model_metrics_dict[f"Precision ({cn})"] = "N/A"
                        current_model_metrics_dict[f"Recall ({cn})"] = "N/A"
                        current_model_metrics_dict[f"F1-Score ({cn})"] = "N/A"
                        current_model_metrics_dict[f"AP@0.5 ({cn})"] = "N/A"
            
            all_metrics_data_list.append(current_model_metrics_dict)
            print(f"Metrics for {model_label} successfully extracted.")

        except Exception as e:
            error_msg_detail = f"Error during evaluation or metrics extraction for {model_label} ({model_path}): {str(e)}"
            print(f"❌ {error_msg_detail}")
            log_message("Error", error_msg_detail, ERROR_LOG)
            import traceback
            log_message("Traceback", traceback.format_exc(), ERROR_LOG)
            current_model_metrics_dict["mAP50 (Overall)"] = "ERROR" # Tandai sebagai error
            all_metrics_data_list.append(current_model_metrics_dict)

    if all_metrics_data_list:
        log_message("Info", f"Writing all extracted metrics to {RESULTS_CSV_FILE}")
        
        # Dapatkan semua header yang mungkin dari data yang dikumpulkan untuk memastikan semua kolom ada
        fieldnames_from_data = set()
        for row_dict in all_metrics_data_list:
            fieldnames_from_data.update(row_dict.keys())
        
        # Buat header standar dan tambahkan header per kelas
        base_headers = [
            "Model Label", "mAP50 (Overall)", "mAP50-95 (Overall)",
            "Precision (Overall)", "Recall (Overall)", "F1-Score (Overall)",
            "Inference Speed (ms/img)", "Preprocessing Speed (ms/img)", "NMS Speed (ms/img)"
        ]
        per_class_metric_types = ["Precision", "Recall", "F1-Score", "AP@0.5"]
        dynamic_class_headers = []
        if class_names: # Hanya jika class_names berhasil dibaca
            for class_n in class_names:
                for metric_t in per_class_metric_types:
                    dynamic_class_headers.append(f"{metric_t} ({class_n})")
        
        # Gabungkan dan urutkan header, pastikan hanya header yang ada di data yang digunakan
        final_ordered_headers = base_headers + sorted(dynamic_class_headers)
        # Filter lagi untuk memastikan hanya header yang benar-benar ada di 'fieldnames_from_data'
        final_ordered_headers = [h for h in final_ordered_headers if h in fieldnames_from_data]
        # Tambahkan header sisa yang mungkin tidak ada di urutan standar (jarang terjadi jika logika benar)
        for remaining_header in sorted(list(fieldnames_from_data - set(final_ordered_headers))):
            if remaining_header not in final_ordered_headers: # Hindari duplikasi
                final_ordered_headers.append(remaining_header)


        try:
            with open(RESULTS_CSV_FILE, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=final_ordered_headers, extrasaction='ignore') # extrasaction='ignore' jika ada key di data yg tdk di fieldnames
                writer.writeheader()
                for row_data_dict in all_metrics_data_list:
                    writer.writerow(row_data_dict)
            log_message("Info", f"Successfully wrote metrics to {RESULTS_CSV_FILE}")
            print(f"\n✅ All model metrics saved to {RESULTS_CSV_FILE}")
        except Exception as e:
            log_message("Error", f"Failed to write metrics to CSV {RESULTS_CSV_FILE}: {e}", ERROR_LOG)
            print(f"❌ Failed to write metrics to CSV {RESULTS_CSV_FILE}: {e}")

    print("\n--- Finished All Evaluations ---")
    log_message("Status", "Validation process finished.")