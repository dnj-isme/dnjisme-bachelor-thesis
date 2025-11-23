import pandas as pd
import os

TRAINED_MODELS_BASE_DIR = "./runs"
OUTPUT_SUMMARY_CSV = "./train_recap.csv" # Nama file output baru
PRIMARY_METRIC_FOR_BEST = 'metrics/mAP50(B)' # Atau 'metrics/mAP50-95(B)'

all_best_epoch_data = []

EXPECTED_COLUMNS = [
    'epoch', 'time',
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
    'metrics/precision(B)', 'metrics/recall(B)',
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
    'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
    'lr/pg0', 'lr/pg1', 'lr/pg2'
]

# Membuat header CSV dengan urutan yang diinginkan
csv_headers = ["Model_Run"] + EXPECTED_COLUMNS


print(f"Mencari model di: {TRAINED_MODELS_BASE_DIR}")
print(f"Metrik utama untuk epoch terbaik: {PRIMARY_METRIC_FOR_BEST}")
print(f"File output akan disimpan sebagai: {OUTPUT_SUMMARY_CSV}")


for folder_name in os.listdir(TRAINED_MODELS_BASE_DIR):
    potential_run_dir = os.path.join(TRAINED_MODELS_BASE_DIR, folder_name)
    if os.path.isdir(potential_run_dir):
        results_csv_path = os.path.join(potential_run_dir, "results.csv")
        if os.path.exists(results_csv_path):
            print(f"\nMemproses: {results_csv_path}")
            try:
                df_log = pd.read_csv(results_csv_path)
                # Membersihkan spasi ekstra dari nama kolom
                df_log.columns = df_log.columns.str.strip()

                if PRIMARY_METRIC_FOR_BEST not in df_log.columns:
                    print(f"  ⚠️ Metrik utama '{PRIMARY_METRIC_FOR_BEST}' tidak ditemukan. Skipping.")
                    continue
                
                # Hapus baris di mana metrik utama adalah NaN atau non-numerik (jika ada)
                # sebelum mencari idxmax() untuk menghindari error jika ada nilai 'inf' atau string
                df_log_cleaned = df_log.copy()
                if df_log_cleaned[PRIMARY_METRIC_FOR_BEST].dtype == 'object':
                    # Coba konversi ke numerik, error akan menjadi NaN
                    df_log_cleaned[PRIMARY_METRIC_FOR_BEST] = pd.to_numeric(df_log_cleaned[PRIMARY_METRIC_FOR_BEST], errors='coerce')
                
                df_log_cleaned = df_log_cleaned.dropna(subset=[PRIMARY_METRIC_FOR_BEST])

                if df_log_cleaned.empty:
                    print(f"  ⚠️ Tidak ada data valid untuk metrik utama '{PRIMARY_METRIC_FOR_BEST}' setelah pembersihan. Skipping.")
                    continue

                best_epoch_idx_cleaned = df_log_cleaned[PRIMARY_METRIC_FOR_BEST].idxmax()
                # Dapatkan indeks asli dari DataFrame original
                best_epoch_idx_original = df_log_cleaned.loc[best_epoch_idx_cleaned].name
                best_epoch_row = df_log.loc[best_epoch_idx_original]


                model_data = {"Model_Run": folder_name}
                for col in EXPECTED_COLUMNS:
                    model_data[col] = best_epoch_row.get(col, 'N/A') # Ambil nilai jika ada, jika tidak 'N/A'
                
                all_best_epoch_data.append(model_data)
                print(f"  ✅ Data epoch terbaik (Epoch: {model_data.get('epoch', 'N/A')}) diekstrak untuk {folder_name}")

            except Exception as e:
                print(f"  ❌ Error memproses {results_csv_path}: {e}")
                import traceback
                print(traceback.format_exc())


if all_best_epoch_data:
    # Membuat DataFrame dari list of dictionaries
    summary_df = pd.DataFrame(all_best_epoch_data)
    
    # Mengatur urutan kolom sesuai dengan csv_headers
    # Filter kolom yang ada di summary_df untuk menghindari KeyError jika ada kolom di csv_headers yang tidak ada di data
    actual_columns_order = [col for col in csv_headers if col in summary_df.columns]
    summary_df = summary_df[actual_columns_order]

    try:
        summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, float_format='%.5f') # Format float untuk konsistensi
        print(f"\n👍 Ringkasan performa epoch terbaik pada validation set disimpan di: {OUTPUT_SUMMARY_CSV}")
    except Exception as e:
        print(f"\n❌ Gagal menyimpan CSV: {e}")
else:
    print("\n🤔 Tidak ada data yang berhasil diekstrak untuk ringkasan.")