from ultralytics import YOLO
import os
import time
import psutil
import threading
import datetime

ERROR_LOG = "error.log"
STATUS_LOG = "status.log"
ITER = 1

def log_message(message_type, description):
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  log_entry = f"[{timestamp}] - {message_type} - {description}\n"
  with open(STATUS_LOG, "a") as log_file:
    log_file.write(log_entry)

def log_resources():
  while resource_logging:
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    log_message("Resource", f"CPU: {cpu_usage}% | Memory: {memory_usage}%")
    time.sleep(600)

if __name__ == "__main__":
  resource_logging = True
  
  log_message("Status", "Process start")
  
  resource_thread = threading.Thread(target=log_resources, daemon=True)
  resource_thread.start()

  model_dir = os.path.abspath("./yolo")
  model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
  model_names = [os.path.splitext(f)[0] for f in model_files]

  print("Detected model files:", model_files)

  for i in range(len(model_files)):
    model_path = os.path.join(model_dir, model_files[i])
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
      print(f"⚠️ Model file {model_files[i]} does NOT exist! Skipping...")
      continue
    
    print(model_path)
    model = YOLO(model_path)
    name = f"Run#{ITER:03d}_vanilla_{model_names[i]}"

    model_type = model_names[i][-1]
    batch_size = 4
    if model_type in ['n', 't']:
      batch_size = 32
    elif model_type == 's':
      batch_size = 16
    elif model_type == 'm':
      batch_size = 8
    
    log_message("Training", f"Starting the training data in model {model_names[i]}")
    
    try:
      model.train(
        name=name,
        data="./datasets/data.yaml",
        epochs=300,
        imgsz=640,
        batch=batch_size,
        project="runs",
        weight_decay=0.001,
        lr0=0.005,
        lrf=0.00005,
        warmup_epochs=3.0,
        warmup_bias_lr=0.01,
        cos_lr=True,
        optimizer="AdamW",
        patience=20
      )
      log_message("Training", f"Finishing the training data in model {model_type}")
    except Exception as e:
      error_msg = f"{model_names[i]} - {str(e)}"
      with open(ERROR_LOG, "a") as err_file:
        err_file.write(f"[{datetime.datetime.now()}] - {error_msg}\n")
      log_message("Error", error_msg)
  
  resource_logging = False
  log_message("Status", "Process finish")