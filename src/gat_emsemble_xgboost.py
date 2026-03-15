import os
import re
import json
import shutil
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Dataset
import logging
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


# Paths
OUTPUT_DIR = "outputs_ensemble_gat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ERROR_DIR = os.path.join(OUTPUT_DIR, "error_analysis")
os.makedirs(ERROR_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "log_ensemble.txt")
RUN_RESULTS_JSON = os.path.join(OUTPUT_DIR, "run_results_ensemble.json")
FINAL_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "final_summary_ensemble.json")
CONFUSION_NPY = os.path.join(OUTPUT_DIR, "confusion_ensemble.npy")
PREDICTION_TABLE_CSV = os.path.join(OUTPUT_DIR, "prediction_table.csv")

RCPD_ANNOTATION_FIX = "rcpd_annotation_fix.csv"
GRAPH_DATA = "graph_data.pt"
RCPD_GRAPHS_PROCESSED = "rcpd_graphs_processed.pt"

POSE_IMAGE_DIR = "pose_graph_images"
SCENE_IMAGE_DIR = "scene_graph_images"

POSE_IMAGE_PREFIX = "pose_graph"
SCENE_IMAGE_PREFIX = "scene_graph"


# Experiment config
FEATURE_TYPE = "stacked_fusion"
LAMBDA_SCENE = 0.8
EPOCHS = 100
PATIENCE = 20
LR = 2.5e-4
WEIGHT_DECAY = 1.4e-5
BATCH_SIZE = 8
N_FOLDS = 5
N_RUNS = 1
NUM_CLASSES = 2
BASE_SEED = 42

POSE_HIDDEN_DIM = 128
POSE_HEADS = 8
POSE_LAYERS = 3
POSE_DROPOUT = 0.3
NUM_KEYPOINTS = 17
KP_EMB_DIM = 16

SCENE_HIDDEN_DIM = 128
SCENE_HEADS = 8
SCENE_LAYERS = 3
SCENE_DROPOUT = 0.3
NUM_TOKENS = 151
NUM_RELATIONS = 51
SCENE_EMB_DIM = 256
SCENE_BBOX_DIM = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def clear_error_files():
    os.makedirs(ERROR_DIR, exist_ok=True)
    for fname in ["TP.txt", "TN.txt", "FP.txt", "FN.txt"]:
        fpath = os.path.join(ERROR_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("filename,prediction,ground_truth,probability,fold,seed,run,feature_type\n")


def save_error_files(tp_lines, tn_lines, fp_lines, fn_lines):
    mapping = {
        "TP.txt": tp_lines,
        "TN.txt": tn_lines,
        "FP.txt": fp_lines,
        "FN.txt": fn_lines,
    }
    for fname, lines in mapping.items():
        if not lines:
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


def build_pose_image_candidates(filename: str, gt_label: int) -> List[str]:
    image_id = extract_image_id(filename)
    if gt_label == 1:
        return [f"{POSE_IMAGE_PREFIX}_{image_id}_csam.png"]
    return [
        f"{POSE_IMAGE_PREFIX}_{image_id}_safe.png",
        f"{POSE_IMAGE_PREFIX}_{image_id}_adult.png",
    ]


def build_scene_image_candidates(filename: str, gt_label: int) -> List[str]:
    image_id = extract_image_id(filename)
    if gt_label == 1:
        return [f"{SCENE_IMAGE_PREFIX}_{image_id}_csam.png"]
    return [
        f"{SCENE_IMAGE_PREFIX}_{image_id}_safe.png",
        f"{SCENE_IMAGE_PREFIX}_{image_id}_adult.png",
    ]


def copy_first_existing(candidates: List[str], src_dir: str, dst_dir: str, label: str):
    found = False
    for name in candidates:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))
            found = True
            break
    if not found:
        logger.warning(f"{label} image not found. Tried: {candidates}")


def copy_images_to_bucket(
    filename: str,
    gt_label: int,
    pred_label: int,
    fold: int,
    run_id: int,
):
    if pred_label == 1 and gt_label == 1:
        bucket = "TP"
    elif pred_label == 0 and gt_label == 0:
        bucket = "TN"
    elif pred_label == 1 and gt_label == 0:
        bucket = "FP"
    else:
        bucket = "FN"

    base_dir = os.path.join(ERROR_DIR, f"run{run_id:02d}", f"fold{fold:02d}", bucket)
    pose_dir = os.path.join(base_dir, "pose")
    scene_dir = os.path.join(base_dir, "scene")
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(scene_dir, exist_ok=True)

    pose_candidates = build_pose_image_candidates(filename, gt_label)
    copy_first_existing(pose_candidates, POSE_IMAGE_DIR, pose_dir, "Pose")


def train_epoch_pose(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for pose_batch, scene_batch in loader:
        pose_batch = pose_batch.to(device)
        optimizer.zero_grad()
        logits = model(pose_batch)
        loss = criterion(logits, pose_batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pose_batch.num_graphs
    return total_loss / len(loader.dataset)


def train_epoch_scene(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for pose_batch, scene_batch in loader:
        scene_batch = scene_batch.to(device)
        optimizer.zero_grad()
        logits = model(scene_batch)
        loss = criterion(logits, scene_batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * scene_batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_pose(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for pose_batch, scene_batch in loader:
        pose_batch = pose_batch.to(device)
        logits = model(pose_batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(pose_batch.y.cpu().tolist())
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
def evaluate_scene(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for pose_batch, scene_batch in loader:
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


class PoseEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = POSE_HIDDEN_DIM,
        heads: int = POSE_HEADS,
        num_layers: int = POSE_LAYERS,
        dropout: float = POSE_DROPOUT,
        num_keypoints: int = NUM_KEYPOINTS,
        kp_emb_dim: int = KP_EMB_DIM,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, \
            f"POSE hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"

        self.keypoint_emb = nn.Embedding(num_keypoints, kp_emb_dim)
        self.input_proj = nn.Linear(in_channels + kp_emb_dim, hidden_dim)

        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        kp_ids = torch.arange(x.size(0), device=x.device) % self.keypoint_emb.num_embeddings
        x = torch.cat([x, self.keypoint_emb(kp_ids)], dim=-1)
        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x = x + self.dropout(norm(conv(x, edge_index).relu()))

        return global_mean_pool(x, batch)


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


class PoseOnlyModel(nn.Module):
    def __init__(
        self,
        pose_in_channels: int,
        pose_hidden_dim: int = POSE_HIDDEN_DIM,
        out_channels: int = NUM_CLASSES,
        dropout: float = POSE_DROPOUT,
    ):
        super().__init__()
        self.pose_encoder = PoseEncoder(pose_in_channels, hidden_dim=pose_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(pose_hidden_dim, pose_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(pose_hidden_dim, pose_hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(pose_hidden_dim // 2, out_channels),
        )

    def forward(self, pose_data: Data, return_embedding: bool = False):
        g_pose = self.pose_encoder(pose_data)
        logits = self.classifier(g_pose)
        if return_embedding:
            return logits, g_pose
        return logits


class SceneOnlyModel(nn.Module):
    def __init__(
        self,
        scene_hidden_dim: int = SCENE_HIDDEN_DIM,
        out_channels: int = NUM_CLASSES,
        dropout: float = SCENE_DROPOUT,
    ):
        super().__init__()
        self.scene_encoder = SceneEncoder(hidden_dim=scene_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(scene_hidden_dim, scene_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(scene_hidden_dim, scene_hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(scene_hidden_dim // 2, out_channels),
        )

    def forward(self, scene_data: Data, return_embedding: bool = False):
        g_scene = self.scene_encoder(scene_data)
        logits = self.classifier(g_scene)
        if return_embedding:
            return logits, g_scene
        return logits


class DualGraphDataset(Dataset):
    def __init__(self, pose_graphs: List[Data], scene_graphs: List[Data]):
        assert len(pose_graphs) == len(scene_graphs)
        self.pose_graphs = pose_graphs
        self.scene_graphs = scene_graphs

    def __len__(self):
        return len(self.pose_graphs)

    def __getitem__(self, idx):
        return self.pose_graphs[idx], self.scene_graphs[idx]


def _worker_init_fn(worker_id: int):
    seed = BASE_SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dual_collate(batch):
    pose_list, scene_list = zip(*batch)
    return Batch.from_data_list(list(pose_list)), Batch.from_data_list(list(scene_list))


def load_dataset() -> Tuple[List[Data], List[Data], np.ndarray, List[str]]:
    df = pd.read_csv(RCPD_ANNOTATION_FIX)
    labels = df["csam"].astype(int).values[:1630]
    filenames = df["filename"].values[:1630]

    pose_graphs = torch.load(GRAPH_DATA, weights_only=False)[:1630]
    scene_graphs = torch.load(RCPD_GRAPHS_PROCESSED, weights_only=False)

    valid_pose, valid_scene, valid_labels, valid_filenames = [], [], [], []
    skipped = 0

    for pose, scene, label, fname in zip(pose_graphs, scene_graphs, labels, filenames):
        pose.x = torch.tensor(pose.x, dtype=torch.float32)
        pose.edge_index = torch.tensor(pose.edge_index, dtype=torch.long)
        pose.y = torch.tensor(label, dtype=torch.long)
        scene.y = torch.tensor(label, dtype=torch.long)

        if pose.x.shape[0] == 0 or pose.edge_index.shape[1] == 0:
            skipped += 1
            continue
        if scene.x.shape[0] == 0 or scene.edge_index.shape[1] == 0:
            skipped += 1
            continue

        valid_pose.append(pose)
        valid_scene.append(scene)
        valid_labels.append(label)
        valid_filenames.append(fname)

    logger.info(f"Skipped {skipped} pairs with empty graphs")
    return valid_pose, valid_scene, np.array(valid_labels), valid_filenames


@torch.no_grad()
def extract_xgb_features(
    pose_model,
    scene_model,
    loader,
    feature_type="embedding",
    lambda_scene=0.8
):
    pose_model.eval()
    scene_model.eval()

    X, y = [], []

    for pose_batch, scene_batch in loader:
        pose_batch = pose_batch.to(device)
        scene_batch = scene_batch.to(device)

        if feature_type == "embedding":
            _, g_pose = pose_model(pose_batch, return_embedding=True)
            _, g_scene = scene_model(scene_batch, return_embedding=True)
            feats = torch.cat([g_pose, g_scene], dim=1)

        elif feature_type == "logits":
            logits_pose = pose_model(pose_batch)
            logits_scene = scene_model(scene_batch)
            feats = torch.cat([logits_pose, logits_scene], dim=1)

        elif feature_type == "probs":
            logits_pose = pose_model(pose_batch)
            logits_scene = scene_model(scene_batch)
            probs_pose = torch.softmax(logits_pose, dim=1)
            probs_scene = torch.softmax(logits_scene, dim=1)
            feats = torch.cat([probs_pose, probs_scene], dim=1)

        elif feature_type == "embedding_logits":
            logits_pose, g_pose = pose_model(pose_batch, return_embedding=True)
            logits_scene, g_scene = scene_model(scene_batch, return_embedding=True)
            feats = torch.cat(
                [
                    g_pose,
                    g_scene,
                    logits_pose[:, 1].unsqueeze(1),
                    logits_scene[:, 1].unsqueeze(1),
                ],
                dim=1,
            )

        elif feature_type == "embedding_probs_weighted":
            logits_pose, g_pose = pose_model(pose_batch, return_embedding=True)
            logits_scene, g_scene = scene_model(scene_batch, return_embedding=True)

            probs_pose = torch.softmax(logits_pose, dim=1)[:, 1].unsqueeze(1)
            probs_scene = torch.softmax(logits_scene, dim=1)[:, 1].unsqueeze(1)
            probs_weighted = lambda_scene * probs_scene + (1.0 - lambda_scene) * probs_pose

            feats = torch.cat(
                [
                    g_pose,
                    g_scene,
                    probs_pose,
                    probs_scene,
                    probs_weighted,
                ],
                dim=1,
            )

        elif feature_type == "stacked_fusion":
            logits_pose, g_pose = pose_model(pose_batch, return_embedding=True)
            logits_scene, g_scene = scene_model(scene_batch, return_embedding=True)

            logit_pose_pos = logits_pose[:, 1].unsqueeze(1)
            logit_scene_pos = logits_scene[:, 1].unsqueeze(1)

            probs_pose = torch.softmax(logits_pose, dim=1)[:, 1].unsqueeze(1)
            probs_scene = torch.softmax(logits_scene, dim=1)[:, 1].unsqueeze(1)

            feats = torch.cat(
                [
                    g_pose,
                    g_scene,
                    logit_pose_pos,
                    logit_scene_pos,
                    probs_pose,
                    probs_scene,
                ],
                dim=1,
            )

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        X.append(feats.cpu().numpy())
        y.append(pose_batch.y.cpu().numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    return X, y


def evaluate_xgb_model(xgb_model, X_test, y_test):
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    return acc, f1, auc, cm, precision, recall, y_pred, y_prob


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


def collect_predictions_and_copy_images(
    filenames_subset: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    fold: int,
    seed: int,
    run_id: int,
    feature_type: str,
):
    prediction_rows = []
    tp_lines, tn_lines, fp_lines, fn_lines = [], [], [], []

    for fname, gt, pred, prob in zip(filenames_subset, y_true, y_pred, y_prob):
        gt = int(gt)
        pred = int(pred)
        prob = float(prob)

        prediction_rows.append({
            "filename": fname,
            "ground_truth": gt,
            "prediction": pred,
            "probability": prob,
            "fold": fold,
            "seed": seed,
            "run": run_id,
            "feature_type": feature_type,
        })

        line = f"{fname},{pred},{gt},{prob:.6f},{fold},{seed},{run_id},{feature_type}"

        if pred == 1 and gt == 1:
            tp_lines.append(line)
        elif pred == 0 and gt == 0:
            tn_lines.append(line)
        elif pred == 1 and gt == 0:
            fp_lines.append(line)
        else:
            fn_lines.append(line)

        copy_images_to_bucket(
            filename=fname,
            gt_label=gt,
            pred_label=pred,
            fold=fold,
            run_id=run_id,
        )

    return prediction_rows, tp_lines, tn_lines, fp_lines, fn_lines


def run_kfold_xgboost_fusion(
    pose_graphs: List[Data],
    scene_graphs: List[Data],
    labels: np.ndarray,
    filenames: List[str],
    seed: int,
    run_id: int,
    feature_type: str = "stacked_fusion",
):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_results = []
    all_prediction_rows = []
    pose_in_channels = pose_graphs[0].x.shape[1]

    for fold, (dev_idx, test_idx) in enumerate(skf.split(pose_graphs, labels), start=1):
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

        train_ds = DualGraphDataset(
            [pose_graphs[i] for i in train_idx],
            [scene_graphs[i] for i in train_idx]
        )
        val_ds = DualGraphDataset(
            [pose_graphs[i] for i in val_idx],
            [scene_graphs[i] for i in val_idx]
        )
        test_ds = DualGraphDataset(
            [pose_graphs[i] for i in test_idx],
            [scene_graphs[i] for i in test_idx]
        )

        test_filenames = [filenames[i] for i in test_idx]
        y_test_true = labels[test_idx]

        g = torch.Generator()
        g.manual_seed(seed + fold)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=dual_collate,
            worker_init_fn=_worker_init_fn,
            generator=g
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=dual_collate,
            worker_init_fn=_worker_init_fn
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=dual_collate,
            worker_init_fn=_worker_init_fn
        )

        pose_model = PoseOnlyModel(pose_in_channels).to(device)
        scene_model = SceneOnlyModel().to(device)

        pose_optimizer = torch.optim.Adam(pose_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scene_optimizer = torch.optim.Adam(scene_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        pose_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            pose_optimizer, T_max=EPOCHS, eta_min=1e-6
        )
        scene_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            scene_optimizer, T_max=EPOCHS, eta_min=1e-6
        )

        train_labels = labels[train_idx]
        neg, pos = (train_labels == 0).sum(), (train_labels == 1).sum()
        class_weights = torch.tensor([1.0 / neg, 1.0 / pos], dtype=torch.float32).to(device)
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_pose_acc = -1.0
        best_scene_acc = -1.0
        best_pose_state = None
        best_scene_state = None
        pose_patience = 0
        scene_patience = 0

        for epoch in range(1, EPOCHS + 1):
            pose_loss = train_epoch_pose(pose_model, train_loader, pose_optimizer, criterion)
            scene_loss = train_epoch_scene(scene_model, train_loader, scene_optimizer, criterion)

            pose_scheduler.step()
            scene_scheduler.step()

            pose_acc, _, _, _, _, _ = evaluate_pose(pose_model, val_loader)
            scene_acc, _, _, _, _, _ = evaluate_scene(scene_model, val_loader)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:>3d} | "
                    f"pose_loss={pose_loss:.4f} val_pose_acc={pose_acc:.4f} | "
                    f"scene_loss={scene_loss:.4f} val_scene_acc={scene_acc:.4f}"
                )

            if pose_acc > best_pose_acc:
                best_pose_acc = pose_acc
                best_pose_state = {k: v.cpu().clone() for k, v in pose_model.state_dict().items()}
                pose_patience = 0
            else:
                pose_patience += 1

            if scene_acc > best_scene_acc:
                best_scene_acc = scene_acc
                best_scene_state = {k: v.cpu().clone() for k, v in scene_model.state_dict().items()}
                scene_patience = 0
            else:
                scene_patience += 1

            if pose_patience >= PATIENCE and scene_patience >= PATIENCE:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best pose val_acc={best_pose_acc:.4f}, best scene val_acc={best_scene_acc:.4f})"
                )
                break

        pose_model.load_state_dict(best_pose_state)
        scene_model.load_state_dict(best_scene_state)

        X_train, y_train = extract_xgb_features(
            pose_model, scene_model, train_loader,
            feature_type=feature_type,
            lambda_scene=LAMBDA_SCENE
        )
        X_val, y_val = extract_xgb_features(
            pose_model, scene_model, val_loader,
            feature_type=feature_type,
            lambda_scene=LAMBDA_SCENE
        )
        X_test, y_test = extract_xgb_features(
            pose_model, scene_model, test_loader,
            feature_type=feature_type,
            lambda_scene=LAMBDA_SCENE
        )

        X_meta_train = np.concatenate([X_train, X_val], axis=0)
        y_meta_train = np.concatenate([y_train, y_val], axis=0)

        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed + fold,
        )

        xgb_model.fit(X_meta_train, y_meta_train)

        acc, f1, auc, cm, prec, rec, y_pred, y_prob = evaluate_xgb_model(xgb_model, X_test, y_test)

        prediction_rows, tp_lines, tn_lines, fp_lines, fn_lines = collect_predictions_and_copy_images(
            filenames_subset=test_filenames,
            y_true=y_test_true,
            y_pred=y_pred,
            y_prob=y_prob,
            fold=fold,
            seed=seed,
            run_id=run_id,
            feature_type=feature_type,
        )

        all_prediction_rows.extend(prediction_rows)
        save_error_files(tp_lines, tn_lines, fp_lines, fn_lines)

        logger.info(
            f"► Final XGBoost fusion ({feature_type}) "
            f"acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  prec={prec:.4f}  rec={rec:.4f}"
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
            "feature_type": feature_type,
            "best_pose_val_acc": float(best_pose_acc),
            "best_scene_val_acc": float(best_scene_acc),
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
    accs = [r["acc_mean"] for r in all_run_summaries]
    f1s = [r["f1_mean"] for r in all_run_summaries]
    aucs = [r["auc_mean"] for r in all_run_summaries]
    precs = [r["prec_mean"] for r in all_run_summaries]
    recs = [r["rec_mean"] for r in all_run_summaries]

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


def run_multiple_experiments(
    pose_graphs: List[Data],
    scene_graphs: List[Data],
    labels: np.ndarray,
    filenames: List[str],
    feature_type: str = "stacked_fusion",
    n_runs: int = N_RUNS
):
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
        fold_results, prediction_rows = run_kfold_xgboost_fusion(
            pose_graphs,
            scene_graphs,
            labels,
            filenames,
            seed=run_seed,
            run_id=run_idx + 1,
            feature_type=feature_type
        )

        all_prediction_rows.extend(prediction_rows)
        run_summary = summarize_fold_results(fold_results)
        print_run_summary(run_idx + 1, fold_results)

        all_run_results.append({
            "run": run_idx + 1,
            "seed": run_seed,
            "feature_type": feature_type,
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
    logger.info(f"Feature type: {FEATURE_TYPE}")
    logger.info(f"Reading annotations from: {RCPD_ANNOTATION_FIX}")
    logger.info(f"Reading pose graphs from: {GRAPH_DATA}")
    logger.info(f"Reading scene graphs from: {RCPD_GRAPHS_PROCESSED}")
    logger.info(f"Reading pose images from: {POSE_IMAGE_DIR}")
    logger.info(f"Reading scene images from: {SCENE_IMAGE_DIR}")
    logger.info(f"Saving outputs to: {OUTPUT_DIR}")

    pose_graphs, scene_graphs, labels, filenames = load_dataset()
    logger.info(
        f"Loaded {len(pose_graphs)} valid pairs  |  "
        f"positives={labels.sum()}  negatives={(labels == 0).sum()}"
    )

    all_run_results, all_run_summaries, all_prediction_rows = run_multiple_experiments(
        pose_graphs,
        scene_graphs,
        labels,
        filenames,
        feature_type=FEATURE_TYPE,
        n_runs=N_RUNS
    )

    prediction_df = pd.DataFrame(all_prediction_rows)
    prediction_df.to_csv(PREDICTION_TABLE_CSV, index=False)

    print_final_summary(all_run_summaries, all_run_results)