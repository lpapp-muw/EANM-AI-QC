#!/usr/bin/env python3
"""
EANM-AI-QC — unified QML (tabular) + QCNN (NIfTI PET ± mask) training/inference

Key properties:
- real quantum circuits evaluated via PennyLane simulator (default.qubit)
- explicit printed diagnostics during training:
  * TRAIN confusion matrix + sensitivity/specificity/PPV/NPV
  * TEST confusion matrix + sensitivity/specificity/PPV/NPV
  * per-test-case predictions (pred vs true) only
- robust model serialization:
  * avoids pickling custom preprocessing classes under __main__
  * backward-compatible load for older pickles referencing __main__.TabularPreprocess
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import dump, load as joblib_load

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


# -------------------------
# Warning handling (numpy 2.x compatibility)
# -------------------------

try:
    from numpy.exceptions import ComplexWarning as _ComplexWarning  # numpy>=2
except Exception:
    try:
        from numpy import ComplexWarning as _ComplexWarning  # numpy<2
    except Exception:
        _ComplexWarning = None

if _ComplexWarning is not None:
    warnings.filterwarnings("ignore", category=_ComplexWarning)


# -------------------------
# Logging
# -------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


# -------------------------
# Metrics helpers
# -------------------------

def binary_confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if den != 0 else float("nan")

    sens = safe_div(tp, tp + fn)  # recall, TPR
    spec = safe_div(tn, tn + fp)  # TNR
    ppv = safe_div(tp, tp + fp)   # precision
    npv = safe_div(tn, tn + fn)

    return {
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
    }


def print_binary_report(
    tag: str,
    y_true: np.ndarray,
    prob1: np.ndarray,
    threshold: float = 0.5,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    prob1 = np.asarray(prob1).astype(float)
    y_pred = (prob1 >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, prob1)) if len(np.unique(y_true)) == 2 else float("nan")
    stats = binary_confusion_stats(y_true, y_pred)

    log(f"{tag} acc={acc:.4f} auc={auc:.4f}")
    log(f"{tag} confusion_matrix [[TN FP],[FN TP]] = [[{int(stats['tn'])} {int(stats['fp'])}],[{int(stats['fn'])} {int(stats['tp'])}]]")
    log(f"{tag} sensitivity={stats['sensitivity']:.4f} specificity={stats['specificity']:.4f} ppv={stats['ppv']:.4f} npv={stats['npv']:.4f}")


# -------------------------
# Generic utilities
# -------------------------

def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_nifti_file(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def strip_nii_ext(filename: str) -> str:
    f = filename
    if f.lower().endswith(".nii.gz"):
        return f[:-7]
    if f.lower().endswith(".nii"):
        return f[:-4]
    return Path(f).stem


def pad_to_power_of_two(vec: np.ndarray, pad_value: float = 0.0) -> Tuple[np.ndarray, int, int]:
    vec = np.asarray(vec, dtype=np.float64).ravel()
    if vec.size < 1:
        vec = np.zeros((1,), dtype=np.float64)
    n = int(math.ceil(math.log2(vec.size)))
    pad_len = 1 << n
    if vec.size == pad_len:
        return vec, n, pad_len
    out = np.full((pad_len,), pad_value, dtype=np.float64)
    out[: vec.size] = vec
    return out, n, pad_len


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(vec))
    return vec / max(n, eps)


def _to_py_scalar(v: Any) -> Any:
    """Convert numpy scalar types (np.int64, np.float32, np.str_, ...) to Python scalars for JSON."""
    try:
        import numpy as _np
        if isinstance(v, _np.generic):
            return v.item()
    except Exception:
        pass
    return v


class BinaryLabelMapper:
    def __init__(self) -> None:
        self.classes_: List[Any] = []
        self.to_int_: Dict[Any, int] = {}
        self.to_label_: Dict[int, Any] = {}

    def fit(self, y: Union[np.ndarray, List[Any]]) -> "BinaryLabelMapper":
        uniq = pd.unique(pd.Series(y))
        if len(uniq) != 2:
            raise ValueError(f"Binary classification required; got {len(uniq)} unique labels: {list(uniq)}")
        self.classes_ = [_to_py_scalar(v) for v in sorted(list(uniq), key=lambda x: str(x))]
        self.to_int_ = {self.classes_[0]: 0, self.classes_[1]: 1}
        self.to_label_ = {0: self.classes_[0], 1: self.classes_[1]}
        return self

    def transform(self, y: Union[np.ndarray, List[Any]]) -> np.ndarray:
        return np.array([self.to_int_[v] for v in y], dtype=np.int64)

    def inverse_transform(self, y01: Union[np.ndarray, List[int]]) -> List[Any]:
        return [self.to_label_[int(v)] for v in y01]

    def to_json(self) -> Dict[str, Any]:
        return {"classes": [_to_py_scalar(self.classes_[0]), _to_py_scalar(self.classes_[1])]}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "BinaryLabelMapper":
        m = BinaryLabelMapper()
        m.classes_ = list(d["classes"])
        m.to_int_ = {m.classes_[0]: 0, m.classes_[1]: 1}
        m.to_label_ = {0: m.classes_[0], 1: m.classes_[1]}
        return m


# -------------------------
# Tabular preprocessing + models
# -------------------------

@dataclass
class TabularPreprocess:
    columns_: List[str]
    imputer: SimpleImputer
    scaler: MinMaxScaler
    pca: PCA
    n_qubits: int

    @staticmethod
    def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df, drop_first=True)

    @staticmethod
    def fit(X: pd.DataFrame, n_qubits: int, seed: int) -> "TabularPreprocess":
        Xo = TabularPreprocess._one_hot(X.copy())
        cols = list(Xo.columns)
        arr = Xo.to_numpy(dtype=np.float64)

        imputer = SimpleImputer(strategy="median")
        arr_i = imputer.fit_transform(arr)

        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        arr_s = scaler.fit_transform(arr_i)

        n_features = 1 << n_qubits
        pca = PCA(n_components=n_features, random_state=seed)
        _ = pca.fit_transform(arr_s)

        return TabularPreprocess(columns_=cols, imputer=imputer, scaler=scaler, pca=pca, n_qubits=n_qubits)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xo = self._one_hot(X.copy())
        Xo = Xo.reindex(columns=self.columns_, fill_value=0.0)
        arr = Xo.to_numpy(dtype=np.float64)

        arr = self.imputer.transform(arr)
        arr = self.scaler.transform(arr)
        arr = self.pca.transform(arr)

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms

    def to_pickle_dict(self) -> Dict[str, Any]:
        return {
            "columns_": self.columns_,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "pca": self.pca,
            "n_qubits": self.n_qubits,
        }

    @staticmethod
    def from_pickle_dict(d: Dict[str, Any]) -> "TabularPreprocess":
        return TabularPreprocess(
            columns_=d["columns_"],
            imputer=d["imputer"],
            scaler=d["scaler"],
            pca=d["pca"],
            n_qubits=int(d["n_qubits"]),
        )


# Backward-compatibility: older model.joblib may reference "__main__.TabularPreprocess"
# When qnm_qai.py was executed as a script, class module name becomes __main__.
# Register this class on the current __main__ module so unpickling can succeed.
try:
    import __main__ as _main
    if not hasattr(_main, "TabularPreprocess"):
        _main.TabularPreprocess = TabularPreprocess
except Exception:
    pass


class AmplitudeKernelSVM_PL:
    """
    PennyLane amplitude-encoded quantum kernel:
      k(x,z) = |<x|z>|^2
    and classical SVC on the precomputed kernel.
    """

    def __init__(self, n_qubits: int = 3, shots: Optional[int] = None, seed: int = 0) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.seed = seed

        self.prep: Optional[TabularPreprocess] = None
        self.labelmap: Optional[BinaryLabelMapper] = None
        self.svc: Optional[SVC] = None
        self.X_ref_: Optional[np.ndarray] = None

        self.last_eval_: Optional[Dict[str, Any]] = None  # train/test split diagnostics

        self._pl_ready = False
        self._dev = None
        self._kernel_qnode = None

    def _init_pl(self) -> None:
        if self._pl_ready:
            return
        import pennylane as qml

        self._dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        wires = list(range(self.n_qubits))

        @qml.qnode(self._dev)
        def kernel_qnode(x1: np.ndarray, x2: np.ndarray) -> Any:
            qml.AmplitudeEmbedding(features=x1, wires=wires, normalize=True)
            return qml.expval(qml.Projector(x2, wires=wires))

        self._kernel_qnode = kernel_qnode
        self._pl_ready = True

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray, symmetric: bool = False) -> np.ndarray:
        self._init_pl()
        K = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)

        if symmetric:
            for i in range(A.shape[0]):
                K[i, i] = float(self._kernel_qnode(A[i], B[i]))
                for j in range(i + 1, B.shape[0]):
                    v = float(self._kernel_qnode(A[i], B[j]))
                    K[i, j] = v
                    K[j, i] = v
            return K

        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                K[i, j] = float(self._kernel_qnode(A[i], B[j]))
        return K

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ids: Optional[List[str]] = None,
        test_size: float = 0.2,
        max_samples: int = 200,
    ) -> Dict[str, float]:
        seed_everything(self.seed)

        if ids is None:
            ids = [str(i) for i in range(len(y))]

        # optional subsample for kernel feasibility
        if len(y) > max_samples:
            idx = np.random.choice(len(y), size=max_samples, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
            ids = [ids[i] for i in idx.tolist()]

        self.labelmap = BinaryLabelMapper().fit(y.tolist())
        y01 = self.labelmap.transform(y.tolist())

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y01, ids,
            test_size=test_size, random_state=self.seed, stratify=y01
        )

        self.prep = TabularPreprocess.fit(X_train, n_qubits=self.n_qubits, seed=self.seed)
        X_train_e = self.prep.transform(X_train)
        X_test_e = self.prep.transform(X_test)

        log(f"[TABULAR] PennyLane kernel matrices: train={X_train_e.shape[0]} test={X_test_e.shape[0]} qubits={self.n_qubits}")
        K_train = self._kernel_matrix(X_train_e, X_train_e, symmetric=True)
        K_test = self._kernel_matrix(X_test_e, X_train_e, symmetric=False)

        self.svc = SVC(kernel="precomputed", probability=True, random_state=self.seed)
        self.svc.fit(K_train, y_train)
        self.X_ref_ = X_train_e

        prob_train = self.svc.predict_proba(K_train)[:, 1]
        prob_test = self.svc.predict_proba(K_test)[:, 1]
        pred_train = (prob_train >= 0.5).astype(int)
        pred_test = (prob_test >= 0.5).astype(int)

        # store for diagnostics printing
        self.last_eval_ = {
            "ids_train": list(ids_train),
            "ids_test": list(ids_test),
            "y_train": y_train,
            "y_test": y_test,
            "prob1_train": prob_train,
            "prob1_test": prob_test,
            "pred_train": pred_train,
            "pred_test": pred_test,
        }

        return {
            "acc": float(accuracy_score(y_test, pred_test)),
            "auc": float(roc_auc_score(y_test, prob_test)) if len(np.unique(y_test)) == 2 else float("nan"),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.prep is None or self.svc is None or self.X_ref_ is None:
            raise RuntimeError("Model not fitted/loaded.")
        X_e = self.prep.transform(X)
        K = self._kernel_matrix(X_e, self.X_ref_, symmetric=False)
        return self.svc.predict_proba(K)

    def save_dir(self, out_dir: Path) -> None:
        if self.prep is None or self.svc is None or self.labelmap is None or self.X_ref_ is None:
            raise RuntimeError("Nothing to save.")
        out_dir = ensure_dir(out_dir)

        meta = {
            "model_type": "tabular_pl_amplitude_kernel_svm",
            "n_qubits": self.n_qubits,
            "shots": self.shots,
            "seed": self.seed,
            "labelmap": self.labelmap.to_json(),
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # IMPORTANT: store preprocess as a plain dict to avoid __main__ pickle issues
        payload = {
            "prep_dict": self.prep.to_pickle_dict(),
            "svc": self.svc,
            "X_ref": self.X_ref_,
        }
        dump(payload, out_dir / "model.joblib")

    @staticmethod
    def load_dir(model_dir: Path) -> "AmplitudeKernelSVM_PL":
        meta = json.loads((Path(model_dir) / "metadata.json").read_text())
        payload = joblib_load(Path(model_dir) / "model.joblib")

        m = AmplitudeKernelSVM_PL(
            n_qubits=int(meta["n_qubits"]),
            shots=meta.get("shots", None),
            seed=int(meta.get("seed", 0)),
        )
        m.labelmap = BinaryLabelMapper.from_json(meta["labelmap"])

        # Backward/forward compatible payload handling
        if isinstance(payload, dict) and "prep_dict" in payload:
            m.prep = TabularPreprocess.from_pickle_dict(payload["prep_dict"])
            m.svc = payload["svc"]
            m.X_ref_ = payload["X_ref"]
        else:
            # older payload likely stored TabularPreprocess instance as "prep"
            # and may use different keys
            m.prep = payload.get("prep", None)
            m.svc = payload.get("svc", None)
            m.X_ref_ = payload.get("X_ref", None)
            if m.prep is None or m.svc is None or m.X_ref_ is None:
                raise RuntimeError("Unsupported older model.joblib format. Retrain the model.")
        return m


# -------------------------
# NIfTI PET±mask loader
# -------------------------

@dataclass
class NiftiPreprocessConfig:
    target_shape: Optional[Tuple[int, int, int]]
    pad_len: int
    n_qubits: int
    intensity_norm: str
    pet_pattern: str
    mask_pattern: Optional[str]


def _try_read_nifti_array(path: Path) -> np.ndarray:
    try:
        import nibabel as nib  # type: ignore
        img = nib.load(str(path))
        return np.asarray(img.get_fdata(), dtype=np.float64)
    except Exception:
        try:
            import SimpleITK as sitk  # type: ignore
            img = sitk.ReadImage(str(path))
            return sitk.GetArrayFromImage(img).astype(np.float64)
        except Exception as e:
            raise RuntimeError("Cannot read NIfTI. Install nibabel (recommended) or SimpleITK.") from e


def _resample_to_shape(image: np.ndarray, target_shape: Tuple[int, int, int], is_mask: bool) -> np.ndarray:
    from scipy.ndimage import zoom  # type: ignore
    if image.shape == target_shape:
        return image
    zf = (
        target_shape[0] / image.shape[0],
        target_shape[1] / image.shape[1],
        target_shape[2] / image.shape[2],
    )
    order = 0 if is_mask else 1
    out = zoom(image, zoom=zf, order=order)
    if is_mask:
        out = (out > 0.5).astype(np.float64)
    return out.astype(np.float64)


def parse_label_from_pet_filename(pet_path: Path) -> str:
    stem = strip_nii_ext(pet_path.name)
    tokens = stem.split("_")
    for t in reversed(tokens):
        if t.isdigit():
            return t
    for t in reversed(tokens):
        if re.search(r"\d", t):
            return t
    raise ValueError(f"Cannot parse label from PET filename: {pet_path.name}")


def case_id_from_filename(p: Path, drop_tokens: Tuple[str, ...] = ("pet", "mask", "seg", "segmentation", "roi")) -> str:
    stem = strip_nii_ext(p.name)
    tokens = [t for t in stem.split("_") if t.strip() != ""]
    tokens_clean = [t for t in tokens if t.lower() not in set(drop_tokens)]
    if len(tokens_clean) >= 2 and tokens_clean[-1].isdigit():
        tokens_clean = tokens_clean[:-1]
    return "_".join(tokens_clean) if tokens_clean else stem


def collect_pet_mask_pairs(root: Path, pet_pattern: str, mask_pattern: Optional[str]) -> List[Tuple[Path, Optional[Path]]]:
    pet_files = sorted([p for p in root.rglob("*") if p.is_file() and is_nifti_file(p) and p.match(pet_pattern)])
    if len(pet_files) == 0:
        pet_files = sorted([p for p in root.rglob("*") if p.is_file() and is_nifti_file(p) and ("mask" not in p.name.lower())])

    mask_map: Dict[str, Path] = {}
    if mask_pattern is not None:
        mask_files = sorted([p for p in root.rglob("*") if p.is_file() and is_nifti_file(p) and p.match(mask_pattern)])
        for m in mask_files:
            mask_map[case_id_from_filename(m)] = m

    pairs: List[Tuple[Path, Optional[Path]]] = []
    for pet in pet_files:
        key = case_id_from_filename(pet)
        pairs.append((pet, mask_map.get(key, None)))

    n_masks = sum(1 for _, m in pairs if m is not None)
    log(f"[NIFTI] root={root} pet_files={len(pet_files)} paired_masks={n_masks} mask_pattern={mask_pattern}")
    return pairs


def build_nifti_matrix(
    pairs: List[Tuple[Path, Optional[Path]]],
    target_shape: Optional[Tuple[int, int, int]],
    intensity_norm: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], int, int]:
    imgs_flat: List[np.ndarray] = []
    labels: List[str] = []
    ids: List[str] = []

    for pet_path, mask_path in pairs:
        img = _try_read_nifti_array(pet_path)

        if mask_path is not None:
            m = _try_read_nifti_array(mask_path)
            if target_shape is not None:
                img = _resample_to_shape(img, target_shape, is_mask=False)
                m = _resample_to_shape(m, target_shape, is_mask=True)
            if img.shape != m.shape:
                raise ValueError(f"PET/mask shape mismatch: {pet_path.name} vs {mask_path.name}: {img.shape} vs {m.shape}")
            img = np.where(m > 0, img, 0.0)  # ROI keep
        else:
            if target_shape is not None:
                img = _resample_to_shape(img, target_shape, is_mask=False)

        if intensity_norm == "zscore_nonzero":
            nz = img[np.isfinite(img) & (img != 0)]
            if nz.size > 0:
                mu = float(nz.mean())
                sd = float(nz.std() + 1e-8)
                img = (img - mu) / sd
            else:
                img = np.zeros_like(img, dtype=np.float64)
        elif intensity_norm == "none":
            pass
        else:
            raise ValueError("Unsupported intensity_norm. Use: none | zscore_nonzero")

        vec = img.ravel().astype(np.float64)
        imgs_flat.append(vec)

        labels.append(parse_label_from_pet_filename(pet_path))
        ids.append(case_id_from_filename(pet_path))

    max_len = max(v.size for v in imgs_flat)
    tmp = []
    for v in imgs_flat:
        if v.size < max_len:
            v2 = np.zeros((max_len,), dtype=np.float64)
            v2[: v.size] = v
            tmp.append(v2)
        else:
            tmp.append(v)

    _, n_qubits, pad_len = pad_to_power_of_two(np.zeros((max_len,), dtype=np.float64), pad_value=0.0)
    X = np.zeros((len(tmp), pad_len), dtype=np.float64)
    for i, v in enumerate(tmp):
        v_pad = np.zeros((pad_len,), dtype=np.float64)
        v_pad[: v.size] = v
        X[i] = l2_normalize(v_pad)

    log(f"[NIFTI] flatten/pad pad_len={pad_len} qubits={n_qubits} target_shape={target_shape} intensity_norm={intensity_norm}")
    return X, np.array(labels), ids, n_qubits, pad_len


def load_nifti_dataset(
    root: Path,
    pet_pattern: str,
    mask_pattern: Optional[str],
    target_shape: Optional[Tuple[int, int, int]],
    intensity_norm: str,
    seed: int,
    test_size: float,
) -> Dict[str, Any]:
    train_dir = root / "Train"
    test_dir = root / "Test"

    if train_dir.exists() and test_dir.exists():
        pairs_train = collect_pet_mask_pairs(train_dir, pet_pattern, mask_pattern)
        pairs_test = collect_pet_mask_pairs(test_dir, pet_pattern, mask_pattern)

        Xtr, ytr_raw, id_tr, n_qubits_tr, pad_len_tr = build_nifti_matrix(pairs_train, target_shape=target_shape, intensity_norm=intensity_norm)
        Xte, yte_raw, id_te, n_qubits_te, pad_len_te = build_nifti_matrix(pairs_test, target_shape=target_shape, intensity_norm=intensity_norm)

        n_qubits = max(n_qubits_tr, n_qubits_te)
        pad_len = 1 << n_qubits

        if pad_len_tr != pad_len:
            Xtr2 = np.zeros((Xtr.shape[0], pad_len), dtype=np.float64)
            Xtr2[:, :pad_len_tr] = Xtr
            Xtr = np.apply_along_axis(l2_normalize, 1, Xtr2)
        if pad_len_te != pad_len:
            Xte2 = np.zeros((Xte.shape[0], pad_len), dtype=np.float64)
            Xte2[:, :pad_len_te] = Xte
            Xte = np.apply_along_axis(l2_normalize, 1, Xte2)

        log(f"[NIFTI] split=Train/Test train={Xtr.shape[0]} test={Xte.shape[0]}")
        return {
            "X_train": Xtr, "y_train_raw": ytr_raw, "ids_train": id_tr,
            "X_test": Xte, "y_test_raw": yte_raw, "ids_test": id_te,
            "n_qubits": n_qubits, "pad_len": pad_len,
        }

    # flat layout -> internal split
    pairs_all = collect_pet_mask_pairs(root, pet_pattern, mask_pattern)
    X, y_raw, ids, n_qubits, pad_len = build_nifti_matrix(pairs_all, target_shape=target_shape, intensity_norm=intensity_norm)

    seed_everything(seed)
    idx = np.arange(len(y_raw))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y_raw)

    log(f"[NIFTI] split=internal train={len(idx_tr)} test={len(idx_te)}")
    return {
        "X_train": X[idx_tr], "y_train_raw": y_raw[idx_tr], "ids_train": [ids[i] for i in idx_tr],
        "X_test": X[idx_te], "y_test_raw": y_raw[idx_te], "ids_test": [ids[i] for i in idx_te],
        "n_qubits": n_qubits, "pad_len": pad_len,
    }


def load_nifti_for_inference(
    root: Path,
    pet_pattern: str,
    mask_pattern: Optional[str],
    target_shape: Optional[Tuple[int, int, int]],
    intensity_norm: str,
    pad_len: int,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    pairs = collect_pet_mask_pairs(root, pet_pattern, mask_pattern)
    X, y_raw, ids, n_qubits, pad_len_found = build_nifti_matrix(pairs, target_shape=target_shape, intensity_norm=intensity_norm)

    if pad_len_found > pad_len:
        raise ValueError(f"Input requires pad_len {pad_len_found} > model pad_len {pad_len}. Resample inputs or retrain.")
    if pad_len_found < pad_len:
        X2 = np.zeros((X.shape[0], pad_len), dtype=np.float64)
        X2[:, :pad_len_found] = X
        X = np.apply_along_axis(l2_normalize, 1, X2)

    return X, ids, y_raw


# -------------------------
# QCNN variants
# -------------------------

class QCNN_MUW_PL:
    def __init__(self, n_layers: Optional[int] = None, shots: Optional[int] = None, seed: int = 0) -> None:
        self.n_layers = n_layers
        self.shots = shots
        self.seed = seed

        self.n_qubits: Optional[int] = None
        self.pad_len: Optional[int] = None
        self.labelmap: Optional[BinaryLabelMapper] = None

        self.w_kernel = None
        self.w_last = None
        self._pl_ready = False
        self._dev = None
        self._qnode = None
        self._n_final: Optional[int] = None

    @staticmethod
    def _auto_layers(n_qubits: int, min_final_wires: int = 2) -> int:
        n = n_qubits
        layers = 0
        while n > min_final_wires:
            n = (n + 1) // 2
            layers += 1
        return max(layers, 1)

    @staticmethod
    def _convolutional_layer(qml, weights_15: Any, wires: List[int], skip_first_layer: bool) -> None:
        n_wires = len(wires)
        if n_wires < 3:
            return
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    w2 = wires[indx + 1]
                    if indx % 2 == 0 and (not skip_first_layer):
                        qml.U3(*weights_15[:3], wires=w)
                        qml.U3(*weights_15[3:6], wires=w2)
                    qml.IsingXX(weights_15[6], wires=[w, w2])
                    qml.IsingYY(weights_15[7], wires=[w, w2])
                    qml.IsingZZ(weights_15[8], wires=[w, w2])
                    qml.U3(*weights_15[9:12], wires=w)
                    qml.U3(*weights_15[12:15], wires=w2)

    @staticmethod
    def _pooling_layer(qml, weights_3: Any, wires: List[int]) -> None:
        if len(wires) < 2:
            return
        for idx in range(1, len(wires), 2):
            ctrl = wires[idx]
            tgt = wires[idx - 1]
            qml.ctrl(qml.U3, control=ctrl)(*weights_3, wires=tgt)

    def _init_pl(self) -> None:
        if self._pl_ready:
            return
        import pennylane as qml
        import pennylane.numpy as pnp

        if self.n_qubits is None:
            raise RuntimeError("n_qubits not set.")
        self._dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)

        n_layers = self.n_layers if self.n_layers is not None else self._auto_layers(self.n_qubits, 2)

        n_active = self.n_qubits
        for _ in range(n_layers):
            n_active = (n_active + 1) // 2
        self._n_final = n_active

        @qml.qnode(self._dev, interface="autograd")
        def qnode(features: Any, w_kernel: Any, w_last: Any) -> Any:
            wires0 = list(range(self.n_qubits))
            qml.AmplitudeEmbedding(features=features, wires=wires0, normalize=True)

            active = wires0
            for layer in range(n_layers):
                self._convolutional_layer(qml, w_kernel[layer][:15], active, skip_first_layer=(layer != 0))
                self._pooling_layer(qml, w_kernel[layer][15:], active)
                active = active[::2]

            expected = (4 ** len(active)) - 1
            if int(pnp.size(w_last)) != int(expected):
                raise ValueError(f"w_last wrong size. expected={expected}, got={pnp.size(w_last)}")
            qml.ArbitraryUnitary(w_last, wires=active)

            return qml.expval(qml.Projector([1], wires=[active[0]]))

        self._qnode = qnode
        self._pl_ready = True

    def fit(
        self,
        X: np.ndarray,
        y01: np.ndarray,
        labelmap: BinaryLabelMapper,
        epochs: int,
        lr: float,
        batch_size: Optional[int],
    ) -> None:
        seed_everything(self.seed)

        width = int(X.shape[1])
        n_qubits = int(round(math.log2(width)))
        if (1 << n_qubits) != width:
            raise ValueError("X must have width 2^n.")
        self.n_qubits = n_qubits
        self.pad_len = width
        self.labelmap = labelmap

        import pennylane as qml
        import pennylane.numpy as pnp

        self._init_pl()
        assert self._n_final is not None

        n_layers = self.n_layers if self.n_layers is not None else self._auto_layers(self.n_qubits, 2)
        log(f"[QCNN/MUW] qubits={self.n_qubits} layers={n_layers} final_wires={self._n_final} epochs={epochs} lr={lr}")

        self.w_kernel = pnp.array(np.random.normal(0, 0.5, size=(n_layers, 18)), requires_grad=True)
        last_len = (4 ** self._n_final) - 1
        self.w_last = pnp.array(np.random.normal(0, 0.5, size=(last_len,)), requires_grad=True)

        def bce(p: Any, y: Any) -> Any:
            eps = 1e-8
            return -(y * pnp.log(p + eps) + (1 - y) * pnp.log(1 - p + eps))

        opt = qml.AdamOptimizer(stepsize=lr)

        N = X.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size > N:
            batch_size = N

        for _ in range(epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, batch_size):
                b = idx[start : start + batch_size]
                xb = X[b]
                yb = y01[b]

                def cost(wk: Any, wl: Any) -> Any:
                    losses = []
                    for i in range(xb.shape[0]):
                        p = self._qnode(xb[i], wk, wl)
                        losses.append(bce(p, yb[i]))
                    return pnp.mean(pnp.stack(losses))

                self.w_kernel, self.w_last = opt.step(cost, self.w_kernel, self.w_last)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w_kernel is None or self.w_last is None:
            raise RuntimeError("Model not fitted/loaded.")
        self._init_pl()
        p1 = np.array([float(self._qnode(X[i], self.w_kernel, self.w_last)) for i in range(X.shape[0])])
        return np.stack([1.0 - p1, p1], axis=1)

    def save_dir(self, out_dir: Path, preprocess_cfg: NiftiPreprocessConfig) -> None:
        if self.n_qubits is None or self.pad_len is None or self.labelmap is None:
            raise RuntimeError("Nothing to save.")
        out_dir = ensure_dir(out_dir)
        meta = {
            "model_type": "nifti_pl_qcnn_muw",
            "seed": self.seed,
            "shots": self.shots,
            "n_qubits": self.n_qubits,
            "pad_len": self.pad_len,
            "n_layers": int(self.w_kernel.shape[0]) if self.w_kernel is not None else None,
            "labelmap": self.labelmap.to_json(),
            "preprocess": {
                "target_shape": list(preprocess_cfg.target_shape) if preprocess_cfg.target_shape is not None else None,
                "intensity_norm": preprocess_cfg.intensity_norm,
                "pet_pattern": preprocess_cfg.pet_pattern,
                "mask_pattern": preprocess_cfg.mask_pattern,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        np.savez(out_dir / "weights.npz", w_kernel=np.array(self.w_kernel, dtype=np.float64), w_last=np.array(self.w_last, dtype=np.float64))

    @staticmethod
    def load_dir(model_dir: Path) -> "QCNN_MUW_PL":
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "metadata.json").read_text())
        w = np.load(model_dir / "weights.npz")
        import pennylane.numpy as pnp
        m = QCNN_MUW_PL(n_layers=int(meta.get("n_layers", 1)), shots=meta.get("shots", None), seed=int(meta.get("seed", 0)))
        m.n_qubits = int(meta["n_qubits"])
        m.pad_len = int(meta["pad_len"])
        m.labelmap = BinaryLabelMapper.from_json(meta["labelmap"])
        m.w_kernel = pnp.array(w["w_kernel"], requires_grad=False)
        m.w_last = pnp.array(w["w_last"], requires_grad=False)
        return m


class QCNN_Alt_PL:
    def __init__(self, n_layers: Optional[int] = None, dense_layers: int = 1, shots: Optional[int] = None, seed: int = 0) -> None:
        self.n_layers = n_layers
        self.dense_layers = dense_layers
        self.shots = shots
        self.seed = seed

        self.n_qubits: Optional[int] = None
        self.pad_len: Optional[int] = None
        self.labelmap: Optional[BinaryLabelMapper] = None

        self.w_kernel = None
        self.w_dense = None
        self._pl_ready = False
        self._dev = None
        self._qnode = None
        self._n_final: Optional[int] = None

    @staticmethod
    def _auto_layers(n_qubits: int, min_final_wires: int = 2) -> int:
        n = n_qubits
        layers = 0
        while n > min_final_wires:
            n = (n + 1) // 2
            layers += 1
        return max(layers, 1)

    @staticmethod
    def _convolutional_layer(qml, weights_15: Any, wires: List[int], skip_first_layer: bool) -> None:
        n_wires = len(wires)
        if n_wires < 3:
            return
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    w2 = wires[indx + 1]
                    if indx % 2 == 0 and (not skip_first_layer):
                        qml.U3(*weights_15[:3], wires=w)
                        qml.U3(*weights_15[3:6], wires=w2)
                    qml.IsingXX(weights_15[6], wires=[w, w2])
                    qml.IsingYY(weights_15[7], wires=[w, w2])
                    qml.IsingZZ(weights_15[8], wires=[w, w2])
                    qml.U3(*weights_15[9:12], wires=w)
                    qml.U3(*weights_15[12:15], wires=w2)

    @staticmethod
    def _pooling_layer(qml, weights_3: Any, wires: List[int]) -> None:
        if len(wires) < 2:
            return
        for idx in range(1, len(wires), 2):
            ctrl = wires[idx]
            tgt = wires[idx - 1]
            qml.ctrl(qml.Rot, control=ctrl)(weights_3[0], weights_3[1], weights_3[2], wires=tgt)

    def _init_pl(self) -> None:
        if self._pl_ready:
            return
        import pennylane as qml

        if self.n_qubits is None:
            raise RuntimeError("n_qubits not set.")
        self._dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)

        n_layers = self.n_layers if self.n_layers is not None else self._auto_layers(self.n_qubits, 2)

        n_active = self.n_qubits
        for _ in range(n_layers):
            n_active = (n_active + 1) // 2
        self._n_final = n_active

        @qml.qnode(self._dev, interface="autograd")
        def qnode(features: Any, w_kernel: Any, w_dense: Any) -> Any:
            wires0 = list(range(self.n_qubits))
            qml.AmplitudeEmbedding(features=features, wires=wires0, normalize=True)

            active = wires0
            for layer in range(n_layers):
                self._convolutional_layer(qml, w_kernel[layer][:15], active, skip_first_layer=(layer != 0))
                self._pooling_layer(qml, w_kernel[layer][15:], active)
                active = active[::2]

            qml.templates.StronglyEntanglingLayers(w_dense, wires=active)
            return qml.expval(qml.Projector([1], wires=[active[0]]))

        self._qnode = qnode
        self._pl_ready = True

    def fit(self, X: np.ndarray, y01: np.ndarray, labelmap: BinaryLabelMapper, epochs: int, lr: float, batch_size: Optional[int]) -> None:
        seed_everything(self.seed)

        width = int(X.shape[1])
        n_qubits = int(round(math.log2(width)))
        if (1 << n_qubits) != width:
            raise ValueError("X must have width 2^n.")
        self.n_qubits = n_qubits
        self.pad_len = width
        self.labelmap = labelmap

        import pennylane as qml
        import pennylane.numpy as pnp

        self._init_pl()
        assert self._n_final is not None

        n_layers = self.n_layers if self.n_layers is not None else self._auto_layers(self.n_qubits, 2)
        log(f"[QCNN/ALT] qubits={self.n_qubits} layers={n_layers} final_wires={self._n_final} dense_layers={self.dense_layers} epochs={epochs} lr={lr}")

        self.w_kernel = pnp.array(np.random.normal(0, 0.5, size=(n_layers, 18)), requires_grad=True)
        self.w_dense = pnp.array(np.random.normal(0, 0.5, size=(self.dense_layers, self._n_final, 3)), requires_grad=True)

        def bce(p: Any, y: Any) -> Any:
            eps = 1e-8
            return -(y * pnp.log(p + eps) + (1 - y) * pnp.log(1 - p + eps))

        opt = qml.AdamOptimizer(stepsize=lr)

        N = X.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size > N:
            batch_size = N

        for _ in range(epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, batch_size):
                b = idx[start : start + batch_size]
                xb = X[b]
                yb = y01[b]

                def cost(wk: Any, wd: Any) -> Any:
                    losses = []
                    for i in range(xb.shape[0]):
                        p = self._qnode(xb[i], wk, wd)
                        losses.append(bce(p, yb[i]))
                    return pnp.mean(pnp.stack(losses))

                self.w_kernel, self.w_dense = opt.step(cost, self.w_kernel, self.w_dense)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w_kernel is None or self.w_dense is None:
            raise RuntimeError("Model not fitted/loaded.")
        self._init_pl()
        p1 = np.array([float(self._qnode(X[i], self.w_kernel, self.w_dense)) for i in range(X.shape[0])])
        return np.stack([1.0 - p1, p1], axis=1)

    def save_dir(self, out_dir: Path, preprocess_cfg: NiftiPreprocessConfig) -> None:
        if self.n_qubits is None or self.pad_len is None or self.labelmap is None:
            raise RuntimeError("Nothing to save.")
        out_dir = ensure_dir(out_dir)
        meta = {
            "model_type": "nifti_pl_qcnn_alt",
            "seed": self.seed,
            "shots": self.shots,
            "n_qubits": self.n_qubits,
            "pad_len": self.pad_len,
            "n_layers": int(self.w_kernel.shape[0]) if self.w_kernel is not None else None,
            "dense_layers": int(self.dense_layers),
            "labelmap": self.labelmap.to_json(),
            "preprocess": {
                "target_shape": list(preprocess_cfg.target_shape) if preprocess_cfg.target_shape is not None else None,
                "intensity_norm": preprocess_cfg.intensity_norm,
                "pet_pattern": preprocess_cfg.pet_pattern,
                "mask_pattern": preprocess_cfg.mask_pattern,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        np.savez(out_dir / "weights.npz", w_kernel=np.array(self.w_kernel, dtype=np.float64), w_dense=np.array(self.w_dense, dtype=np.float64))

    @staticmethod
    def load_dir(model_dir: Path) -> "QCNN_Alt_PL":
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "metadata.json").read_text())
        w = np.load(model_dir / "weights.npz")
        import pennylane.numpy as pnp
        m = QCNN_Alt_PL(n_layers=int(meta.get("n_layers", 1)), dense_layers=int(meta.get("dense_layers", 1)), shots=meta.get("shots", None), seed=int(meta.get("seed", 0)))
        m.n_qubits = int(meta["n_qubits"])
        m.pad_len = int(meta["pad_len"])
        m.labelmap = BinaryLabelMapper.from_json(meta["labelmap"])
        m.w_kernel = pnp.array(w["w_kernel"], requires_grad=False)
        m.w_dense = pnp.array(w["w_dense"], requires_grad=False)
        return m


# -------------------------
# Unified load/save + CSV
# -------------------------

def save_metrics(out_dir: Path, metrics: Dict[str, Any]) -> None:
    (Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_metadata(model_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(model_dir) / "metadata.json").read_text())


def load_any_model(model_dir: Union[str, Path]) -> Any:
    model_dir = Path(model_dir)
    meta = load_metadata(model_dir)
    mt = meta["model_type"]
    if mt == "tabular_pl_amplitude_kernel_svm":
        return AmplitudeKernelSVM_PL.load_dir(model_dir)
    if mt == "nifti_pl_qcnn_muw":
        return QCNN_MUW_PL.load_dir(model_dir)
    if mt == "nifti_pl_qcnn_alt":
        return QCNN_Alt_PL.load_dir(model_dir)
    raise ValueError(f"Unknown model_type: {mt}")


def write_predictions_csv(out_path: Path, ids: List[str], prob1: np.ndarray, pred01: np.ndarray, pred_label: Optional[List[Any]] = None, true_label: Optional[List[Any]] = None) -> None:
    df = pd.DataFrame({"id": ids, "prob_1": prob1, "pred_01": pred01})
    if pred_label is not None:
        df["pred_label"] = pred_label
    if true_label is not None:
        df["true_label"] = true_label
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


# -------------------------
# CLI
# -------------------------

def read_tabular_file(path: Path, label_col: Optional[str], require_label: bool) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = pd.read_csv(path, sep="\t") if path.suffix.lower() == ".tsv" else pd.read_csv(path)

    if label_col is None:
        # safest default: use "label" if present; otherwise treat as unlabeled unless require_label
        if "label" in df.columns:
            label_col = "label"
        elif require_label:
            raise ValueError("Training requires labels. Provide a column named 'label' or pass --label-col <name>.")
        else:
            return df, None

    if label_col in df.columns:
        y = df[label_col]
        X = df.drop(columns=[label_col])
        return X, y

    if require_label:
        raise ValueError(f"Label column '{label_col}' not found in {path}.")
    return df, None


def cmd_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    out_dir = ensure_dir(Path(args.out))

    if args.mode == "tabular":
        log(f"[TRAIN][TABULAR] backend={args.backend} out={out_dir} seed={args.seed}")
        X, y = read_tabular_file(Path(args.data), label_col=args.label_col, require_label=True)
        assert y is not None

        ids = [str(i) for i in range(len(y))]  # stable ids for printing
        log(f"[TABULAR] samples={len(y)} features={X.shape[1]} label_col={args.label_col or 'label'}")

        model = AmplitudeKernelSVM_PL(n_qubits=args.n_qubits, shots=args.shots, seed=args.seed)
        metrics = model.fit(X, y, ids=ids, test_size=args.test_size, max_samples=args.max_samples)

        # evaluation prints (train/test) + per-test-case lines
        assert model.last_eval_ is not None and model.labelmap is not None
        lm = model.labelmap
        log(f"[TABULAR] labelmap.classes={lm.classes_} (prob_1 refers to class '{lm.classes_[1]}')")

        le = model.last_eval_
        print_binary_report("[TABULAR][TRAIN]", le["y_train"], le["prob1_train"])
        print_binary_report("[TABULAR][TEST ]", le["y_test"], le["prob1_test"])

        # per-test-case output only
        y_test_label = lm.inverse_transform(le["y_test"])
        pred_test_label = lm.inverse_transform(le["pred_test"])
        log("[TABULAR][TEST_CASES] id prob_1 pred_label true_label")
        for i in range(len(le["ids_test"])):
            log(f"[TABULAR][TEST_CASE] {le['ids_test'][i]:>6}  prob_1={le['prob1_test'][i]:.4f}  pred={pred_test_label[i]}  true={y_test_label[i]}")

        model.save_dir(out_dir)
        save_metrics(out_dir, {"metrics_test": metrics, "labelmap": lm.to_json()})

        log(f"[TABULAR] saved_model_dir={out_dir}")
        log(f"[TABULAR] saved_metrics={out_dir/'metrics.json'}")
        return

    if args.mode == "nifti":
        log(f"[TRAIN][NIFTI] qcnn={args.qcnn} out={out_dir} seed={args.seed}")
        root = Path(args.data)
        target_shape = tuple(args.target_shape) if args.target_shape is not None else None

        mask_pattern = args.mask_pattern
        if mask_pattern is not None and mask_pattern.upper() == "NONE":
            mask_pattern = None

        ds = load_nifti_dataset(
            root=root,
            pet_pattern=args.pet_pattern,
            mask_pattern=mask_pattern,
            target_shape=target_shape,
            intensity_norm=args.intensity_norm,
            seed=args.seed,
            test_size=args.test_size,
        )

        labelmap = BinaryLabelMapper().fit(ds["y_train_raw"].tolist() + ds["y_test_raw"].tolist())
        ytr = labelmap.transform(ds["y_train_raw"].tolist())
        yte = labelmap.transform(ds["y_test_raw"].tolist())

        log(f"[NIFTI] labelmap.classes={labelmap.classes_} (prob_1 refers to class '{labelmap.classes_[1]}')")

        preprocess_cfg = NiftiPreprocessConfig(
            target_shape=target_shape,
            pad_len=int(ds["pad_len"]),
            n_qubits=int(ds["n_qubits"]),
            intensity_norm=args.intensity_norm,
            pet_pattern=args.pet_pattern,
            mask_pattern=mask_pattern,
        )

        if args.qcnn == "muw":
            model = QCNN_MUW_PL(n_layers=args.qcnn_layers, shots=args.shots, seed=args.seed)
            model.fit(ds["X_train"], ytr, labelmap, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

            prob_train = model.predict_proba(ds["X_train"])[:, 1]
            prob_test = model.predict_proba(ds["X_test"])[:, 1]

            print_binary_report("[NIFTI][TRAIN]", ytr, prob_train)
            print_binary_report("[NIFTI][TEST ]", yte, prob_test)

            pred_test = (prob_test >= 0.5).astype(int)
            true_test_label = labelmap.inverse_transform(yte)
            pred_test_label = labelmap.inverse_transform(pred_test)

            log("[NIFTI][TEST_CASES] id prob_1 pred_label true_label")
            for i in range(len(ds["ids_test"])):
                log(f"[NIFTI][TEST_CASE] {ds['ids_test'][i]}  prob_1={prob_test[i]:.4f}  pred={pred_test_label[i]}  true={true_test_label[i]}")

            model.save_dir(out_dir, preprocess_cfg)
            save_metrics(out_dir, {"labelmap": labelmap.to_json()})
            log(f"[NIFTI] saved_model_dir={out_dir}")
            log(f"[NIFTI] saved_metrics={out_dir/'metrics.json'}")
            return

        if args.qcnn == "alt":
            model = QCNN_Alt_PL(n_layers=args.qcnn_layers, dense_layers=args.qcnn_dense_layers, shots=args.shots, seed=args.seed)
            model.fit(ds["X_train"], ytr, labelmap, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

            prob_train = model.predict_proba(ds["X_train"])[:, 1]
            prob_test = model.predict_proba(ds["X_test"])[:, 1]

            print_binary_report("[NIFTI][TRAIN]", ytr, prob_train)
            print_binary_report("[NIFTI][TEST ]", yte, prob_test)

            pred_test = (prob_test >= 0.5).astype(int)
            true_test_label = labelmap.inverse_transform(yte)
            pred_test_label = labelmap.inverse_transform(pred_test)

            log("[NIFTI][TEST_CASES] id prob_1 pred_label true_label")
            for i in range(len(ds["ids_test"])):
                log(f"[NIFTI][TEST_CASE] {ds['ids_test'][i]}  prob_1={prob_test[i]:.4f}  pred={pred_test_label[i]}  true={true_test_label[i]}")

            model.save_dir(out_dir, preprocess_cfg)
            save_metrics(out_dir, {"labelmap": labelmap.to_json()})
            log(f"[NIFTI] saved_model_dir={out_dir}")
            log(f"[NIFTI] saved_metrics={out_dir/'metrics.json'}")
            return

        raise ValueError("Unknown --qcnn. Use: muw | alt")

    raise ValueError("Unknown --mode. Use: tabular | nifti")


def cmd_predict(args: argparse.Namespace) -> None:
    model_dir = Path(args.model)
    meta = load_metadata(model_dir)
    model = load_any_model(model_dir)
    out_path = Path(args.out)

    mt = meta["model_type"]
    log(f"[PREDICT] model_dir={model_dir} model_type={mt}")
    log(f"[PREDICT] data={args.data} out={out_path}")

    if mt == "tabular_pl_amplitude_kernel_svm":
        X, y = read_tabular_file(Path(args.data), label_col=args.label_col, require_label=False)
        prob = model.predict_proba(X)
        pred01 = (prob[:, 1] >= 0.5).astype(int)

        pred_label = model.labelmap.inverse_transform(pred01) if getattr(model, "labelmap", None) is not None else None
        ids = [str(i) for i in range(len(pred01))]

        true_label = None
        if y is not None and getattr(model, "labelmap", None) is not None:
            # compute and print metrics if labels present
            y01 = model.labelmap.transform(y.tolist())
            print_binary_report("[PREDICT][TABULAR][EVAL]", y01, prob[:, 1])
            true_label = y.tolist()

        write_predictions_csv(out_path, ids, prob[:, 1], pred01, pred_label=pred_label, true_label=true_label)
        log(f"[PREDICT] wrote={out_path} n={len(pred01)}")
        try:
            df = pd.read_csv(out_path).head(8)
            log(df.to_string(index=False))
        except Exception:
            pass
        return

    if mt in {"nifti_pl_qcnn_muw", "nifti_pl_qcnn_alt"}:
        pp = meta["preprocess"]
        target_shape = tuple(pp["target_shape"]) if pp.get("target_shape", None) is not None else None

        X, ids, y_raw = load_nifti_for_inference(
            root=Path(args.data),
            pet_pattern=pp.get("pet_pattern", "*PET*.nii*"),
            mask_pattern=pp.get("mask_pattern", None),
            target_shape=target_shape,
            intensity_norm=pp.get("intensity_norm", "zscore_nonzero"),
            pad_len=int(meta["pad_len"]),
        )

        prob = model.predict_proba(X)
        pred01 = (prob[:, 1] >= 0.5).astype(int)
        pred_label = model.labelmap.inverse_transform(pred01) if getattr(model, "labelmap", None) is not None else None

        # if y_raw seems to be binary labels matching labelmap, print eval
        true_label = None
        if getattr(model, "labelmap", None) is not None:
            try:
                y01 = model.labelmap.transform(y_raw.tolist())
                print_binary_report("[PREDICT][NIFTI][EVAL]", y01, prob[:, 1])
                true_label = y_raw.tolist()
            except Exception:
                pass

        write_predictions_csv(out_path, ids, prob[:, 1], pred01, pred_label=pred_label, true_label=true_label)
        log(f"[PREDICT] wrote={out_path} n={len(ids)}")
        try:
            df = pd.read_csv(out_path).head(8)
            log(df.to_string(index=False))
        except Exception:
            pass
        return

    raise ValueError(f"Unsupported model_type for predict: {mt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qnm_qai")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--mode", required=True, choices=["tabular", "nifti"])
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--shots", type=int, default=None)

    # tabular
    t.add_argument("--label-col", default=None)
    t.add_argument("--backend", default="pennylane", choices=["pennylane"])
    t.add_argument("--n-qubits", type=int, default=3)
    t.add_argument("--max-samples", type=int, default=200)

    # nifti
    t.add_argument("--qcnn", default="muw", choices=["muw", "alt"])
    t.add_argument("--pet-pattern", default="*PET*.nii*")
    t.add_argument("--mask-pattern", default="*mask*.nii*")
    t.add_argument("--target-shape", type=int, nargs=3, default=None)
    t.add_argument("--intensity-norm", default="zscore_nonzero", choices=["none", "zscore_nonzero"])
    t.add_argument("--qcnn-layers", type=int, default=None)
    t.add_argument("--qcnn-dense-layers", type=int, default=1)
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--batch-size", type=int, default=None)

    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True)
    pr.add_argument("--data", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--label-col", default=None)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args)
        return
    if args.cmd == "predict":
        cmd_predict(args)
        return
    raise RuntimeError("Unreachable")


if __name__ == "__main__":
    main()
