#!/usr/bin/env python3
"""
medklip_zero_shot_vindr.py

Run MedKLIP zero-shot classification (logic aligned with Sample_zero-shot_Classification_CXR14/test.py)
on VinDr-CXR PNGs with image_labels_test.csv, and compute metrics for whichever VinDr classes
you can map to MedKLIP's concept vocabulary.

Key change vs your current script:
- Do NOT restrict to the 14-class CXR14 subset.
- Infer probabilities for ALL MedKLIP `original_class` concepts, then select only the concepts
  referenced by your DEFAULT_VINDR_TO_MEDKLIP mapping (and present in the VinDr CSV).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml

from models.model_MedKLIP import MedKLIP
from models.tokenization_bert import BertTokenizer

from sklearn.metrics import roc_auc_score, f1_score


# -----------------------------
# Metrics
# -----------------------------
def compute_map_at_k(y_true, scores, k=10):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    N, L = y_true.shape
    ap_list = []

    for i in range(N):
        y = y_true[i]
        s = scores[i]
        pos_idx = np.where(y == 1)[0]
        if len(pos_idx) == 0:
            continue

        order = np.argsort(-s)
        topk = order[:k]

        hits = 0
        precisions = []
        for rank, idx in enumerate(topk, start=1):
            if y[idx] == 1:
                hits += 1
                precisions.append(hits / rank)

        if len(precisions) == 0:
            ap = 0.0
        else:
            denom = min(len(pos_idx), k)
            ap = float(np.sum(precisions) / denom)

        ap_list.append(ap)

    if len(ap_list) == 0:
        return None
    return float(np.mean(ap_list))


def compute_classification_metrics(y_true, scores, label_names, threshold=0.5):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    N, L = y_true.shape
    assert scores.shape == (N, L)

    metrics = {}

    # ----- ROC–AUC -----
    per_label_auc = {}
    auc_values = []
    for j, label in enumerate(label_names):
        y = y_true[:, j]
        if len(np.unique(y)) < 2:
            per_label_auc[label] = None
            continue
        try:
            auc = roc_auc_score(y, scores[:, j])
            per_label_auc[label] = float(auc)
            auc_values.append(auc)
        except ValueError:
            per_label_auc[label] = None

    metrics["per_label_auc"] = per_label_auc
    metrics["macro_auc"] = float(np.mean(auc_values)) if len(auc_values) > 0 else None

    # micro-AUC
    try:
        metrics["micro_auc"] = float(roc_auc_score(y_true.ravel(), scores.ravel()))
    except ValueError:
        metrics["micro_auc"] = None

    # ----- F1 (global threshold) -----
    y_pred = (scores >= threshold).astype(int)

    per_label_f1 = {}
    f1_values = []
    for j, label in enumerate(label_names):
        y = y_true[:, j]
        y_hat = y_pred[:, j]
        if len(np.unique(y)) < 2:
            per_label_f1[label] = None
            continue
        f1 = f1_score(y, y_hat)
        per_label_f1[label] = float(f1)
        f1_values.append(f1)

    metrics["per_label_f1"] = per_label_f1
    metrics["macro_f1"] = float(np.mean(f1_values)) if len(f1_values) > 0 else None
    metrics["micro_f1"] = float(f1_score(y_true.ravel(), y_pred.ravel()))

    # ----- mAP@10 -----
    metrics["map_at_10"] = compute_map_at_k(y_true, scores, k=10)

    return metrics


# -----------------------------
# MedKLIP concept vocabulary (from test.py)
# -----------------------------
original_class = [
    'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
    'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process',
    'abnormality', 'enlarge', 'tip', 'low', 'pneumonia', 'line', 'congestion', 'catheter',
    'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
    'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid',
    'degenerative', 'pacemaker', 'thicken', 'marking', 'scar', 'hyperinflate', 'blunt', 'loss',
    'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure',
    'deformity', 'hernia', 'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding',
    'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
    'tail_abnorm_obs', 'excluded_obs'
]
ORIG_CONCEPT_TO_IDX = {c: i for i, c in enumerate(original_class)}


# -----------------------------
# VinDr->MedKLIP mapping
# VinDr column name -> MedKLIP concept name (must be in original_class)
# -----------------------------
DEFAULT_VINDR_TO_MEDKLIP = {
    "Aortic enlargement": "enlarge",  # check
    "Atelectasis": "atelectasis",
    "Calcification": "calcification",
    "Cardiomegaly": "cardiomegaly",
    # "Clavicle fracture": "fracture",  # not sure if fracture can be used here
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Enlarged PA": "enlarge",  # check
    "ILD": "loss",  # can be opacity too
    "Infiltration": "infiltrate",
    "Lung Opacity": "opacity",
    # "Lung cyst": "absent",  # not present
    "Mediastinal shift": "shift",
    "Nodule/Mass": "nodule",  # can be mass too
    "Pleural effusion": "effusion",
    "Pleural thickening": "thicken",
    "Pneumothorax": "pneumothorax",
    "Pulmonary fibrosis": "loss",
    "Rib fracture": "fracture",
    "Other lesion": "lesion",
    "COPD": "hernia",
    # "Lung tumor": "absent"  # not present
    "Pneumonia": "pneumonia",
    # "Tuberculosis": "absent" # associated to three labels which have better associations with another label
    # "Other disease": "absent",  # vague
    'No finding': "normal"
}


# -----------------------------
# Utilities
# -----------------------------
def resolve_checkpoint_paths(
    model_paths_or_dirs: List[str],
    exts: Tuple[str, ...] = (".pth", ".pt"),
    recursive: bool = False,
) -> List[str]:
    out: List[Path] = []
    for item in model_paths_or_dirs:
        p = Path(item)
        if p.is_dir():
            globber = p.rglob if recursive else p.glob
            for ext in exts:
                out.extend(globber(f"*{ext}"))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {p}")

    uniq = sorted({str(p.resolve()) for p in out})
    if len(uniq) == 0:
        raise FileNotFoundError(
            f"No checkpoints found in inputs={model_paths_or_dirs} exts={exts} recursive={recursive}"
        )
    return uniq


def build_eval_transform():
    # Matches MedKLIP dataset.dataset.py is_train=False
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize,
    ])


def build_eval_concepts_from_mapping(
    df: pd.DataFrame,
    vindr_to_medklip: Dict[str, str],
):
    """
    Returns:
      eval_vindr_cols: VinDr columns we will evaluate (in stable order)
      concept_indices: indices into `original_class` for those columns (same order)
      eval_concepts:   corresponding MedKLIP concept names (same order)
    Also returns unmapped VinDr columns for reporting.
    """
    eval_vindr_cols: List[str] = []
    eval_concepts: List[str] = []
    concept_indices: List[int] = []

    # Stable, explicit order: follow CSV column order (excluding image_id)
    for col in df.columns:
        if col == "image_id":
            continue
        if col not in vindr_to_medklip:
            continue
        concept = vindr_to_medklip[col]
        if concept not in ORIG_CONCEPT_TO_IDX:
            continue
        eval_vindr_cols.append(col)
        eval_concepts.append(concept)
        concept_indices.append(ORIG_CONCEPT_TO_IDX[concept])

    # Report all VinDr columns that could not be evaluated (for transparency)
    unmapped = []
    for col in df.columns:
        if col == "image_id":
            continue
        if col not in vindr_to_medklip:
            unmapped.append(col)
        else:
            concept = vindr_to_medklip[col]
            if concept not in ORIG_CONCEPT_TO_IDX:
                unmapped.append(col)

    if len(eval_vindr_cols) == 0:
        raise RuntimeError(
            "No evaluable VinDr columns found. Ensure DEFAULT_VINDR_TO_MEDKLIP maps VinDr column names "
            "to concept names present in MedKLIP `original_class`."
        )

    return eval_vindr_cols, concept_indices, eval_concepts, unmapped


class VinDrPNGDataset(Dataset):
    """
    Returns:
      {"image": tensor(3,224,224), "label": float32(K,), "image_id": str}

    label is aligned to eval_vindr_cols order (K == number of evaluable VinDr columns).
    """
    def __init__(
        self,
        image_dir: str,
        df: pd.DataFrame,
        image_id_col: str,
        eval_vindr_cols: List[str],
        transform,
    ):
        self.image_dir = Path(image_dir)
        self.df = df.reset_index(drop=True)
        self.image_id_col = image_id_col
        self.eval_vindr_cols = list(eval_vindr_cols)
        self.transform = transform

        # Validate image files exist (fail fast)
        missing = 0
        for iid in self.df[self.image_id_col].astype(str).tolist():
            p = self.image_dir / f"{iid}.png"
            if not p.exists():
                missing += 1
        if missing > 0:
            raise FileNotFoundError(f"[VinDrPNGDataset] Missing {missing} PNG files in {self.image_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row[self.image_id_col])
        img_path = self.image_dir / f"{image_id}.png"

        img = Image.open(img_path).convert("RGB")
        image = self.transform(img)

        y = row[self.eval_vindr_cols].to_numpy(dtype=np.float32)
        return {"image": image, "label": y, "image_id": image_id}


@torch.no_grad()
def infer_one_checkpoint_medklip(
    model,
    dataloader: DataLoader,
    device: torch.device,
    concept_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generalized MedKLIP inference:
      - model produces logits for len(original_class) concepts (2-way logits per concept)
      - we softmax over the 2 classes and take positive prob
      - we select only `concept_indices` to match the evaluable VinDr labels

    Outputs:
      gt:   (N, K) from dataset labels (VinDr columns subset)
      pred: (N, K) MedKLIP probabilities for mapped concepts
      ids:  list[str] image_id order
    """
    gt_list = []
    pred_list = []
    ids: List[str] = []

    model.eval()
    for sample in dataloader:
        image = sample["image"].to(device, non_blocking=True)  # (B,3,224,224)
        label = sample["label"].to(device)                     # (B,K)
        image_ids = sample["image_id"]

        gt_list.append(label.detach().cpu().numpy())
        ids.extend([str(x) for x in image_ids])

        pred_logits = model(image)

        # Exactly as test.py: softmax on flattened 2-way logits then reshape
        probs = F.softmax(pred_logits.reshape(-1, 2), dim=-1).reshape(-1, len(original_class), 2)
        pos = probs[:, :, 1]  # (B, len(original_class))
        pos_sel = pos[:, concept_indices]  # (B, K)

        pred_list.append(pos_sel.detach().cpu().numpy())

    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    return gt, pred, ids


def load_medklip_model(config: dict, checkpoint_path: str, device: torch.device):
    """
    Mirrors Sample_zero-shot_Classification_CXR14/test.py:
      disease_book_tokenizer
      model = MedKLIP(config, disease_book_tokenizer)
      model = DataParallel(model)
      load checkpoint['model']
    """
    json_book = json.load(open(config["disease_book"], "r"))
    disease_book = [json_book[i] for i in json_book]

    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])
    disease_book_tokenizer = tokenizer(
        list(disease_book),
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).to(device)

    model = MedKLIP(config, disease_book_tokenizer)
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    return model


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, required=True, help="Path to MedKLIP_config.yaml")

    ap.add_argument("--model_paths", type=str, nargs="+", required=True,
                    help="Checkpoint file(s) or directory(ies) containing checkpoints")
    ap.add_argument("--ckpt_recursive", action="store_true", help="Recursively search directories for checkpoints")
    ap.add_argument("--ckpt_exts", type=str, default=".pth,.pt", help="Comma-separated extensions to include")

    ap.add_argument("--vindr_root", type=str, required=True, help="Path to vindr_cxr/")
    ap.add_argument("--csv_rel", type=str, default="annotations/image_labels_test.csv")
    ap.add_argument("--img_rel", type=str, default="test")

    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (default auto)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--smoke_n", type=int, default=0, help="If >0, run only first N images")

    ap.add_argument("--label_map_json", type=str, default=None,
                    help="Optional JSON file mapping VinDr column -> MedKLIP concept name (must be in original_class)")

    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, default="vindr_medklip_out")

    args = ap.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    vindr_root = Path(args.vindr_root)
    csv_path = vindr_root / args.csv_rel
    img_dir = vindr_root / args.img_rel
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns:
        raise ValueError(f"VinDr CSV must contain 'image_id'. Columns={list(df.columns)}")

    if args.smoke_n and args.smoke_n > 0:
        df = df.iloc[: args.smoke_n].reset_index(drop=True)

    # Resolve checkpoints
    ckpt_exts = tuple(e.strip() for e in args.ckpt_exts.split(",") if e.strip())
    ckpt_paths = resolve_checkpoint_paths(args.model_paths, exts=ckpt_exts, recursive=args.ckpt_recursive)
    print(f"[Info] Found {len(ckpt_paths)} checkpoint(s)")

    # Label mapping
    vindr_to_medklip = dict(DEFAULT_VINDR_TO_MEDKLIP)
    if args.label_map_json:
        user_map = json.load(open(args.label_map_json, "r"))
        if not isinstance(user_map, dict):
            raise ValueError("label_map_json must be a JSON object mapping VinDr column -> MedKLIP concept string")
        vindr_to_medklip.update(user_map)

    eval_vindr_cols, concept_indices, eval_concepts, unmapped_vindr_cols = build_eval_concepts_from_mapping(
        df=df,
        vindr_to_medklip=vindr_to_medklip,
    )

    print(f"[Info] Evaluable VinDr labels: {len(eval_vindr_cols)}")
    for c, concept in zip(eval_vindr_cols, eval_concepts):
        print(f"       {c} -> {concept}")
    print(f"[Info] Unmappable VinDr labels (cannot evaluate with this checkpoint): {len(unmapped_vindr_cols)}")

    # Device
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Dataset/loader
    transform = build_eval_transform()
    dataset = VinDrPNGDataset(
        image_dir=str(img_dir),
        df=df,
        image_id_col="image_id",
        eval_vindr_cols=eval_vindr_cols,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # preserve CSV order
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save mapping/meta
    (out_dir / "vindr_to_medklip_used.json").write_text(json.dumps(vindr_to_medklip, indent=2) + "\n")
    (out_dir / "eval_vindr_cols.txt").write_text("\n".join(eval_vindr_cols) + "\n")
    (out_dir / "eval_medklip_concepts.txt").write_text("\n".join(eval_concepts) + "\n")
    (out_dir / "unmapped_vindr_cols.txt").write_text("\n".join(unmapped_vindr_cols) + "\n")

    # Run each checkpoint (with caching), then average
    preds_list = []
    gt_ref = None
    ids_ref = None

    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for ckpt in ckpt_paths:
        ckpt = str(Path(ckpt).resolve())
        model_name = Path(ckpt).stem
        cache_path = cache_dir / f"{model_name}.npy"

        if cache_path.exists():
            print(f"[Cache] Loading cached predictions: {cache_path}")
            pred = np.load(cache_path)
            preds_list.append(pred)
            continue

        print(f"[Infer] Loading MedKLIP model: {ckpt}")
        model = load_medklip_model(config, ckpt, device)

        print(f"[Infer] Running inference for {model_name} on {len(dataset)} images")
        gt, pred, ids = infer_one_checkpoint_medklip(
            model=model,
            dataloader=loader,
            device=device,
            concept_indices=concept_indices,
        )

        if gt_ref is None:
            gt_ref = gt
            ids_ref = ids
        else:
            if gt.shape != gt_ref.shape:
                raise RuntimeError(f"gt shape mismatch: {gt.shape} vs {gt_ref.shape}")
            if ids != ids_ref:
                raise RuntimeError("Image ID ordering changed across runs; ensure shuffle=False and deterministic dataset.")
            if pred.shape != preds_list[0].shape:
                raise RuntimeError(f"pred shape mismatch: {pred.shape} vs {preds_list[0].shape}")

        np.save(cache_path, pred.astype(np.float32))
        preds_list.append(pred)

    assert gt_ref is not None and ids_ref is not None

    scores_avg = np.mean(np.stack(preds_list, axis=0), axis=0)  # (N,K)

    # Save outputs aligned to evaluable VinDr columns
    np.save(out_dir / "scores_eval.npy", scores_avg.astype(np.float32))
    np.save(out_dir / "y_true_eval.npy", gt_ref.astype(np.int32))
    (out_dir / "image_ids.txt").write_text("\n".join(ids_ref) + "\n")

    metrics = compute_classification_metrics(
        y_true=gt_ref,
        scores=scores_avg,
        label_names=eval_vindr_cols,  # evaluate by VinDr label names
        threshold=args.threshold,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    meta = {
        "config": str(Path(args.config).resolve()),
        "vindr_root": str(vindr_root.resolve()),
        "csv": str(csv_path.resolve()),
        "img_dir": str(img_dir.resolve()),
        "num_images": int(len(dataset)),
        "num_eval_labels": int(len(eval_vindr_cols)),
        "eval_vindr_cols": eval_vindr_cols,
        "eval_medklip_concepts": eval_concepts,
        "device": device_str,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "smoke_n": int(args.smoke_n),
        "checkpoints": ckpt_paths,
        "ensemble_size": int(len(ckpt_paths)),
        "threshold": float(args.threshold),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[Done] Outputs in: {out_dir}")
    print(f"       scores_eval.npy: {scores_avg.shape} (N,K)")
    print(f"       metrics.json: computed on K={len(eval_vindr_cols)} mapped VinDr labels")


if __name__ == "__main__":
    main()
