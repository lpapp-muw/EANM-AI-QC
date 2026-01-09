echo "[EXAMPLE] Repo root: ${ROOT_DIR}"
echo "[EXAMPLE] Project: EANM-AI-QC"

mkdir -p models outputs demo_data

echo "[EXAMPLE] Tabular dataset: demo_data/tabular/real_train.csv"
if [ ! -f demo_data/tabular/real_train.csv ]; then
  echo "[EXAMPLE] real_train.csv missing -> building from demo_data/tabular/raw/FDB.csv + LDB.csv"
  python3 examples/build_tabular_from_fdb_ldb.py
fi

echo "[EXAMPLE] Tabular dataset: demo_data/tabular/real_train.csv"
if [ ! -f demo_data/tabular/real_train.csv ]; then
  echo "[EXAMPLE] real_train.csv missing -> building from demo_data/tabular/raw/FDB.csv + LDB.csv"
  python3 examples/build_tabular_from_fdb_ldb.py
fi

echo "[EXAMPLE] Generating synthetic NIfTI demo inputs"
python3 examples/make_synthetic_nifti.py
echo
echo "[EXAMPLE] Tabular (PennyLane) — amplitude-kernel SVM"
python3 qnm_qai.py train --mode tabular   --data demo_data/tabular/real_train.csv   --out models/tab_pl   --backend pennylane --n-qubits 3 --test-size 0.25 --max-samples 200

python3 qnm_qai.py predict --model models/tab_pl   --data demo_data/tabular/real_infer.csv   --out outputs/preds_tab_pl.csv

echo "[EXAMPLE] Tabular evaluation predictions (includes true labels)"
python3 qnm_qai.py predict --model models/tab_pl \
  --data demo_data/tabular/real_train.csv \
  --out outputs/preds_tab_pl_eval.csv

echo
echo "[EXAMPLE] NIfTI — QCNN MUW variant (PET+mask pairs)"
python3 qnm_qai.py train --mode nifti   --data demo_data/nifti_masked   --out models/nifti_muw_masked   --qcnn muw --epochs 3 --lr 0.05

python3 qnm_qai.py predict --model models/nifti_muw_masked   --data demo_data/nifti_masked/Test   --out outputs/preds_nifti_muw_masked.csv

echo
echo "[EXAMPLE] NIfTI — QCNN ALT variant (PET-only, masks disabled)"
python3 qnm_qai.py train --mode nifti   --data demo_data/nifti_nomask   --out models/nifti_alt_nomask   --qcnn alt --mask-pattern NONE --epochs 3 --lr 0.05

python3 qnm_qai.py predict --model models/nifti_alt_nomask   --data demo_data/nifti_nomask/Test   --out outputs/preds_nifti_alt_nomask.csv

echo
echo "[EXAMPLE] Outputs:"
ls -lh outputs || true

echo "Done"
