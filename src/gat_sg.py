# Scene-graph branch with N repeated runs and aggregated summary across runs

import os
import re
import json
import shutil
import random
import logging
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

# Experiment config
EPOCHS       = 100
PATIENCE     = 20
LR           = 2.5e-4
WEIGHT_DECAY = 1.4e-5
BATCH_SIZE   = 8
N_FOLDS      = 5
NUM_CLASSES  = 2

BASE_SEED = 42
N_RUNS    = 5

SCENE_HIDDEN_DIM = 128
SCENE_HEADS      = 8
SCENE_LAYERS     = 3
SCENE_DROPOUT    = 0.3
NUM_TOKENS       = 151
NUM_RELATIONS    = 51
SCENE_EMB_DIM    = 256
SCENE_BBOX_DIM   = 32


# Paths
OUTPUT_DIR = "outputs_sg_gat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ERROR_DIR = os.path.join(OUTPUT_DIR, "error_analysis")
os.makedirs(ERROR_DIR, exist_ok=True)

ANNOTATION_FILE  = "rcpd_annotation_fix.csv"
SCENE_GRAPH_FILE = "rcpd_graphs_processed.pt"

SCENE_IMAGE_DIR = "scene_graph_images"
SCENE_IMAGE_PREFIX = "scene_graph"

LOG_FILE            = os.path.join(OUTPUT_DIR, "log_sg.txt")
RUN_RESULTS_JSON    = os.path.join(OUTPUT_DIR, "run_results_sg.json")
FINAL_SUMMARY_JSON  = os.path.join(OUTPUT_DIR, "final_summary_sg.json")
CONFUSION_NPY       = os.path.join(OUTPUT_DIR, "confusion_sg.npy")
PREDICTION_TABLE_CSV = os.path.join(OUTPUT_DIR, "prediction_table.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging():
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


class SceneEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int = NUM_TOKENS,
        num_relations: int = NUM_RELATIONS,
        emb_dim: int = SCENE_EMB_DIM,
        bbox_dim: int = SCENE_BBOX_DIM,
        hidden_dim: int = SCENE_HIDDEN_DIM,
        heads: int = SCENE_HEADS,
        num_layers: int = SCENE_LAYERS,
        dropout: float = SCENE_DROPOUT,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, \
            f"SCENE hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"

        self.tok_emb = nn.Embedding(num_tokens, emb_dim)
        self.bbox_proj = nn.Linear(4, bbox_dim)
        in_dim = emb_dim + bbox_dim

        self.rel_emb = nn.Embedding(num_relations, hidden_dim // heads)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(
            GATv2Conv(
                in_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=hidden_dim // heads,
                dropout=dropout
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(1, num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=hidden_dim // heads,
                    dropout=dropout
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        tok_id = data.x[:, 0].long()
        bbox = data.x[:, 1:].float()

        if bbox.size(1) > 4:
            bbox = bbox[:, :4]
        elif bbox.size(1) < 4:
            bbox = torch.cat([bbox, bbox.new_zeros(bbox.size(0), 4 - bbox.size(1))], dim=1)

        x = torch.cat([self.tok_emb(tok_id), self.bbox_proj(bbox)], dim=-1)
        edge_attr = self.rel_emb(data.edge_attr)

        x = self.dropout(self.norms[0](self.convs[0](x, data.edge_index, edge_attr).relu()))
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            x = x + self.dropout(norm(conv(x, data.edge_index, edge_attr).relu()))

        return global_mean_pool(x, data.batch)


class ClassificationModel(nn.Module):
    def __init__(
        self,
        scene_hidden_dim: int = SCENE_HIDDEN_DIM,
        out_channels: int = NUM_CLASSES,
    ):
        super().__init__()
        self.scene_encoder = SceneEncoder(hidden_dim=scene_hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(scene_hidden_dim, scene_hidden_dim),
            nn.ELU(),
            nn.Dropout(SCENE_DROPOUT),
            nn.Linear(scene_hidden_dim, scene_hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(SCENE_DROPOUT),
            nn.Linear(scene_hidden_dim // 2, out_channels),
        )

    def forward(self, scene_data: Data) -> torch.Tensor:
        g_scene = self.scene_encoder(scene_data)
        return self.mlp(g_scene)


class GraphDataset(Dataset):
    def __init__(self, scene_graphs: List[Data]):
        self.scene_graphs = scene_graphs

    def __len__(self):
        return len(self.scene_graphs)

    def __getitem__(self, idx):
        return self.scene_graphs[idx]


def _worker_init_fn(worker_id: int):
    seed = BASE_SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_error_files():
    for fname in ["TP.txt", "TN.txt", "FP.txt", "FN.txt"]:
        fpath = os.path.join(ERROR_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("filename,prediction,ground_truth,probability,fold,seed,run\n")


def save_error_files(tp_lines, tn_lines, fp_lines, fn_lines):
    mapping = {
        "TP.txt": tp_lines,
        "TN.txt": tn_lines,
        "FP.txt": fp_lines,
        "FN.txt": fn_lines,
    }

    for fname, lines in mapping.items():
        if len(lines) == 0:
            continue
        fpath = os.path.join(ERROR_DIR, fname)
        with open(fpath, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def extract_image_id(filename: str) -> int:
    base = os.path.basename(filename)
    match = re.search(r"img_(\d+)\.jpg$", base)
    if match is None:
        raise ValueError(f"Could not extract image id from filename: {filename}")
    return int(match.group(1))


def build_scene_image_candidates(filename: str, gt_label: int) -> List[str]:
    image_id = extract_image_id(filename)

    if gt_label == 1:
        return [f"{SCENE_IMAGE_PREFIX}_{image_id}_csam.png"]
    else:
        return [
            f"{SCENE_IMAGE_PREFIX}_{image_id}_safe.png",
            f"{SCENE_IMAGE_PREFIX}_{image_id}_adult.png",
        ]


def copy_scene_image_to_bucket(filename: str, gt_label: int, pred_label: int, fold: int, run_id: int):
    if pred_label == 1 and gt_label == 1:
        bucket = "TP"
    elif pred_label == 0 and gt_label == 0:
        bucket = "TN"
    elif pred_label == 1 and gt_label == 0:
        bucket = "FP"
    else:
        bucket = "FN"

    fold_dir = os.path.join(ERROR_DIR, f"run{run_id:02d}", f"fold{fold:02d}", bucket)
    os.makedirs(fold_dir, exist_ok=True)

    candidate_names = build_scene_image_candidates(filename, gt_label)

    found = False
    for scene_image_name in candidate_names:
        src_path = os.path.join(SCENE_IMAGE_DIR, scene_image_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(fold_dir, scene_image_name)
            shutil.copy2(src_path, dst_path)
            found = True
            break

    if not found:
        logger.warning(
            f"Scene graph image not found for {filename}. Tried: {candidate_names}"
        )


def load_dataset() -> Tuple[List[Data], np.ndarray, List[str]]:
    df = pd.read_csv(ANNOTATION_FILE)
    labels = df["csam"].astype(int).values[:1630]
    filenames = df["filename"].values[:1630]
    scene_graphs = torch.load(SCENE_GRAPH_FILE, weights_only=False)

    valid_scene = []
    valid_labels = []
    valid_filenames = []
    skipped = 0

    for scene, label, fname in zip(scene_graphs, labels, filenames):
        scene.y = torch.tensor(label, dtype=torch.long)

        if scene.x.shape[0] == 0 or scene.edge_index.shape[1] == 0:
            skipped += 1
            continue

        valid_scene.append(scene)
        valid_labels.append(label)
        valid_filenames.append(fname)

    logger.info(f"Skipped {skipped} pairs with empty graphs")
    return valid_scene, np.array(valid_labels), valid_filenames


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for scene_batch in loader:
        scene_batch = scene_batch.to(device)
        optimizer.zero_grad()
        logits = model(scene_batch)
        loss = criterion(logits, scene_batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * scene_batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for scene_batch in loader:
        scene_batch = scene_batch.to(device)
        logits = model(scene_batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(scene_batch.y.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return acc, f1, auc, cm, precision, recall


@torch.no_grad()
def evaluate_and_collect_predictions(
    model,
    loader,
    filenames_subset: List[str],
    fold: int,
    seed: int,
    run_id: int,
):
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    prediction_rows = []

    tp_lines, tn_lines, fp_lines, fn_lines = [], [], [], []

    sample_ptr = 0

    for scene_batch in loader:
        scene_batch = scene_batch.to(device)

        logits = model(scene_batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        batch_size = scene_batch.num_graphs

        for i in range(batch_size):
            gt = int(scene_batch.y[i].cpu().item())
            pred = int(preds[i].cpu().item())
            prob = float(probs[i].cpu().item())

            fname = filenames_subset[sample_ptr]
            sample_ptr += 1

            all_labels.append(gt)
            all_preds.append(pred)
            all_probs.append(prob)

            row = {
                "filename": fname,
                "ground_truth": gt,
                "prediction": pred,
                "probability": prob,
                "fold": fold,
                "seed": seed,
                "run": run_id,
            }
            prediction_rows.append(row)

            line = f"{fname},{pred},{gt},{prob:.6f},{fold},{seed},{run_id}"

            if pred == 1 and gt == 1:
                tp_lines.append(line)
            elif pred == 0 and gt == 0:
                tn_lines.append(line)
            elif pred == 1 and gt == 0:
                fp_lines.append(line)
            else:
                fn_lines.append(line)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return (
        acc,
        f1,
        auc,
        cm,
        precision,
        recall,
        prediction_rows,
        tp_lines,
        tn_lines,
        fp_lines,
        fn_lines,
    )


def summarize_fold_results(fold_results: List[Dict]) -> Dict:
    accs = [r["acc"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    precs = [r["prec"] for r in fold_results]
    recs = [r["rec"] for r in fold_results]

    agg_cm = sum(r["cm"] for r in fold_results)

    summary = {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "prec_mean": float(np.mean(precs)),
        "prec_std": float(np.std(precs)),
        "rec_mean": float(np.mean(recs)),
        "rec_std": float(np.std(recs)),
        "agg_cm": agg_cm,
    }
    return summary


def run_kfold(scene_graphs: List[Data], labels: np.ndarray, filenames: List[str], seed: int, run_id: int):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_results = []
    all_prediction_rows = []

    for fold, (dev_idx, test_idx) in enumerate(skf.split(scene_graphs, labels), start=1):
        train_idx, val_idx = train_test_split(
            dev_idx,
            test_size=0.15,
            stratify=labels[dev_idx],
            random_state=seed + fold,
        )

        logger.info('─' * 50)
        logger.info(
            f"Seed {seed} | Fold {fold}/{N_FOLDS}  |  "
            f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
        )
        logger.info('─' * 50)

        train_ds = GraphDataset([scene_graphs[i] for i in train_idx])
        val_ds   = GraphDataset([scene_graphs[i] for i in val_idx])
        test_ds  = GraphDataset([scene_graphs[i] for i in test_idx])

        test_filenames = [filenames[i] for i in test_idx]

        g = torch.Generator()
        g.manual_seed(seed + fold)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            worker_init_fn=_worker_init_fn,
            generator=g
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            worker_init_fn=_worker_init_fn,
            generator=g
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            worker_init_fn=_worker_init_fn,
            generator=g
        )

        model = ClassificationModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )

        train_labels = labels[train_idx]
        neg, pos = (train_labels == 0).sum(), (train_labels == 1).sum()
        class_weights = torch.tensor([1.0 / neg, 1.0 / pos], dtype=torch.float32).to(device)
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_acc = -1.0
        best_state = None
        patience_count = 0

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()
            acc, f1, auc, _, prec, rec = evaluate(model, val_loader)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:>3d}  loss={loss:.4f}  "
                    f"val_acc={acc:.4f}  val_f1={f1:.4f}  val_auc={auc:.4f}  "
                    f"val_prec={prec:.4f}  val_rec={rec:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}  (best val_acc={best_acc:.4f})")
                    break

        model.load_state_dict(best_state)

        (
            acc,
            f1,
            auc,
            cm,
            prec,
            rec,
            prediction_rows,
            tp_lines,
            tn_lines,
            fp_lines,
            fn_lines,
        ) = evaluate_and_collect_predictions(
            model,
            test_loader,
            test_filenames,
            fold=fold,
            seed=seed,
            run_id=run_id,
        )

        all_prediction_rows.extend(prediction_rows)
        save_error_files(tp_lines, tn_lines, fp_lines, fn_lines)

        logger.info(
            f"► Final   acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  "
            f"prec={prec:.4f}  rec={rec:.4f}  (test fold)"
        )
        logger.info(f"► Confusion matrix:\n{cm}")

        fold_results.append({
            "fold": fold,
            "acc": float(acc),
            "f1": float(f1),
            "auc": float(auc) if not np.isnan(auc) else None,
            "cm": cm,
            "prec": float(prec),
            "rec": float(rec),
        })

    return fold_results, all_prediction_rows


def print_run_summary(run_id: int, fold_results: List[Dict]):
    summary = summarize_fold_results(fold_results)
    agg_cm = summary["agg_cm"]

    logger.info('═' * 50)
    logger.info(f"Run {run_id}/{N_RUNS} Summary ({N_FOLDS}-Fold CV)")
    logger.info('═' * 50)
    logger.info(f"Accuracy  : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    logger.info(f"F1 Score  : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    logger.info(f"ROC-AUC   : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    logger.info(f"Precision : {summary['prec_mean']:.4f} ± {summary['prec_std']:.4f}")
    logger.info(f"Recall    : {summary['rec_mean']:.4f} ± {summary['rec_std']:.4f}")
    logger.info(f"Aggregated confusion matrix:\n{agg_cm}")
    logger.info('═' * 50)


def print_final_summary(all_run_summaries: List[Dict], all_run_results: List[Dict]):
    accs  = [r["acc_mean"] for r in all_run_summaries]
    f1s   = [r["f1_mean"] for r in all_run_summaries]
    aucs  = [r["auc_mean"] for r in all_run_summaries]
    precs = [r["prec_mean"] for r in all_run_summaries]
    recs  = [r["rec_mean"] for r in all_run_summaries]

    agg_cm = sum(np.array(r["agg_cm"]) for r in all_run_summaries)
    np.save(CONFUSION_NPY, agg_cm)

    final_summary = {
        "n_runs": N_RUNS,
        "n_folds": N_FOLDS,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "precision_mean": float(np.mean(precs)),
        "precision_std": float(np.std(precs)),
        "recall_mean": float(np.mean(recs)),
        "recall_std": float(np.std(recs)),
        "aggregated_confusion_matrix": agg_cm.tolist(),
    }

    with open(RUN_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_run_results, f, indent=2, ensure_ascii=False)

    with open(FINAL_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    logger.info('█' * 60)
    logger.info(f"Final Summary Across {N_RUNS} Runs")
    logger.info('█' * 60)
    logger.info(f"Accuracy  : {final_summary['accuracy_mean']:.4f} ± {final_summary['accuracy_std']:.4f}")
    logger.info(f"F1 Score  : {final_summary['f1_mean']:.4f} ± {final_summary['f1_std']:.4f}")
    logger.info(f"ROC-AUC   : {final_summary['auc_mean']:.4f} ± {final_summary['auc_std']:.4f}")
    logger.info(f"Precision : {final_summary['precision_mean']:.4f} ± {final_summary['precision_std']:.4f}")
    logger.info(f"Recall    : {final_summary['recall_mean']:.4f} ± {final_summary['recall_std']:.4f}")
    logger.info(f"Aggregated confusion matrix across runs:\n{agg_cm}")
    logger.info(f"Saved → {RUN_RESULTS_JSON}")
    logger.info(f"Saved → {FINAL_SUMMARY_JSON}")
    logger.info(f"Saved → {CONFUSION_NPY}")
    logger.info(f"Saved → {PREDICTION_TABLE_CSV}")
    logger.info('█' * 60)


def run_multiple_experiments(scene_graphs: List[Data], labels: np.ndarray, filenames: List[str], n_runs: int = N_RUNS):
    all_run_results = []
    all_run_summaries = []
    all_prediction_rows = []

    for run_idx in range(n_runs):
        run_seed = BASE_SEED + run_idx

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Starting run {run_idx + 1}/{n_runs} with seed={run_seed}")
        logger.info("=" * 60)

        set_seed(run_seed)
        fold_results, prediction_rows = run_kfold(
            scene_graphs,
            labels,
            filenames,
            seed=run_seed,
            run_id=run_idx + 1,
        )

        all_prediction_rows.extend(prediction_rows)
        run_summary = summarize_fold_results(fold_results)

        print_run_summary(run_idx + 1, fold_results)

        all_run_results.append({
            "run": run_idx + 1,
            "seed": run_seed,
            "fold_results": [
                {
                    **fr,
                    "cm": fr["cm"].tolist()
                } for fr in fold_results
            ],
            "summary": {
                **run_summary,
                "agg_cm": run_summary["agg_cm"].tolist()
            }
        })

        all_run_summaries.append({
            **run_summary,
            "agg_cm": run_summary["agg_cm"]
        })

    return all_run_results, all_run_summaries, all_prediction_rows


if __name__ == "__main__":
    setup_logging()
    clear_error_files()

    logger.info(f"Using device: {device}")
    logger.info(f"Reading annotations from: {ANNOTATION_FILE}")
    logger.info(f"Reading graphs from: {SCENE_GRAPH_FILE}")
    logger.info(f"Reading scene graph images from: {SCENE_IMAGE_DIR}")
    logger.info(f"Saving outputs to: {OUTPUT_DIR}")

    scene_graphs, labels, filenames = load_dataset()
    logger.info(
        f"Loaded {len(scene_graphs)} valid graphs  |  "
        f"positives={labels.sum()}  negatives={(labels == 0).sum()}"
    )

    all_run_results, all_run_summaries, all_prediction_rows = run_multiple_experiments(
        scene_graphs,
        labels,
        filenames,
        n_runs=N_RUNS
    )

    prediction_df = pd.DataFrame(all_prediction_rows)
    prediction_df.to_csv(PREDICTION_TABLE_CSV, index=False)

    print_final_summary(all_run_summaries, all_run_results)