# Skeleton pose branch with N repeated runs and aggregated summary across runs

import os
import json
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import logging
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from typing import List, Tuple

# Paths
OUTPUT_DIR = "outputs_pose_gat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RCPD_ANNOTATION_FIX = "rcpd_annotation_fix.csv"
GRAPH_DATA = "graph_data.pt"
RCPD_GRAPHS_PROCESSED = "rcpd_graphs_processed.pt"

LOG_FILE = os.path.join(OUTPUT_DIR, "log.txt")
RUN_RESULTS_JSON = os.path.join(OUTPUT_DIR, "run_results.json")
FINAL_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "final_summary.json")
CONFUSION_NPY = os.path.join(OUTPUT_DIR, "confusion.npy")

# Experiment config
EPOCHS       = 100
LR           = 3.8e-4
WEIGHT_DECAY = 1.4e-5
BATCH_SIZE   = 8
N_FOLDS      = 5
PATIENCE     = 20
HIDDEN_DIM   = 128
HEADS        = 8
DROPOUT      = 0.3
NUM_CLASSES  = 2
NUM_LAYERS   = 3
NUM_KEYPOINTS = 17
KP_EMB_DIM    = 16

BASE_SEED = 42
N_RUNS    = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configurações para determinismo na GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class ASGRA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = HIDDEN_DIM,
        heads: int = HEADS,
        out_channels: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        num_layers: int = NUM_LAYERS,
        num_keypoints: int = NUM_KEYPOINTS,
        kp_emb_dim: int = KP_EMB_DIM,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"

        self.keypoint_emb = nn.Embedding(num_keypoints, kp_emb_dim)
        self.input_proj = nn.Linear(in_channels + kp_emb_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                )
            )

        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.pool = global_mean_pool

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        kp_ids = torch.arange(x.size(0), device=x.device) % self.keypoint_emb.num_embeddings
        x = torch.cat([x, self.keypoint_emb(kp_ids)], dim=-1)
        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x = x + self.dropout(norm(conv(x, edge_index).relu()))

        g = self.pool(x, batch)
        return self.mlp(g)


def load_dataset() -> Tuple[List[Data], np.ndarray]:
    df = pd.read_csv(RCPD_ANNOTATION_FIX)
    labels = df["csam"].values.astype(int)[:1630]
    graph_list = torch.load(GRAPH_DATA, weights_only=False)[:1630]

    empty_graphs = 0
    valid_graphs = []
    valid_labels = []

    for graph, label in zip(graph_list, labels):
        graph.x = torch.tensor(graph.x, dtype=torch.float32)
        graph.edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        graph.y = torch.tensor(label, dtype=torch.long)

        if graph.x.shape[0] == 0 or graph.edge_index.shape[1] == 0:
            empty_graphs += 1
            continue

        valid_graphs.append(graph)
        valid_labels.append(label)

    logger.info(f"Skipped {empty_graphs} empty graphs")
    return valid_graphs, np.array(valid_labels)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(batch.y.cpu().tolist())
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


def summarize_fold_results(fold_results: List[dict]) -> dict:
    accs  = [r["acc"] for r in fold_results]
    f1s   = [r["f1"] for r in fold_results]
    aucs  = [r["auc"] for r in fold_results]
    precs = [r["prec"] for r in fold_results]
    recs  = [r["rec"] for r in fold_results]

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


def run_kfold(dataset: List[Data], labels: np.ndarray, seed: int):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    fold_results = []
    in_channels = dataset[0].x.shape[1]

    for fold, (dev_idx, test_idx) in enumerate(skf.split(dataset, labels), start=1):
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

        train_loader = DataLoader(
            [dataset[i] for i in train_idx],
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            [dataset[i] for i in val_idx],
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        test_loader = DataLoader(
            [dataset[i] for i in test_idx],
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = ASGRA(
            in_channels,
            HIDDEN_DIM,
            HEADS,
            NUM_CLASSES,
            DROPOUT,
            NUM_LAYERS
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        train_labels = labels[train_idx]
        neg, pos = (train_labels == 0).sum(), (train_labels == 1).sum()
        class_weights = torch.tensor([1.0 / neg, 1.0 / pos], dtype=torch.float32).to(device)
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )

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
        acc, f1, auc, cm, prec, rec = evaluate(model, test_loader)

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

    return fold_results


def print_run_summary(run_id: int, fold_results: List[dict]):
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


def print_final_summary(all_run_summaries: List[dict], all_run_results: List[dict]):
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
    logger.info('█' * 60)


def run_multiple_experiments(dataset: List[Data], labels: np.ndarray, n_runs: int = N_RUNS):
    all_run_results = []
    all_run_summaries = []

    for run_idx in range(n_runs):
        run_seed = BASE_SEED + run_idx
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Starting run {run_idx + 1}/{n_runs} with seed={run_seed}")
        logger.info("=" * 60)

        set_seed(run_seed)
        fold_results = run_kfold(dataset, labels, seed=run_seed)
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

    return all_run_results, all_run_summaries


if __name__ == "__main__":
    setup_logging()
    logger.info(f"Using device: {device}")
    logger.info(f"Reading annotations from: {RCPD_ANNOTATION_FIX}")
    logger.info(f"Reading graphs from: {GRAPH_DATA}")
    logger.info(f"Saving outputs to: {OUTPUT_DIR}")

    dataset, labels = load_dataset()
    logger.info(
        f"Loaded {len(dataset)} graphs  |  "
        f"positives={labels.sum()}  negatives={(labels == 0).sum()}"
    )

    all_run_results, all_run_summaries = run_multiple_experiments(
        dataset,
        labels,
        n_runs=N_RUNS
    )

    print_final_summary(all_run_summaries, all_run_results)