"""
Vanilla baseline: classifier using ALL available features (no acquisition policy).

For each sample, every available feature at each timestep is used.
Results are appended to vanilla.csv (same column format as LAFA_ACTORS - all.csv).

Usage:
    ACTOR_DATASET=klg    python evaluate_vanilla.py
    ACTOR_DATASET=womac  python evaluate_vanilla.py
    ACTOR_DATASET=cheears python evaluate_vanilla.py
"""
import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import AUROC
from torchmetrics.classification import AveragePrecision

from config import (
    DATA_FOLDER, CLASSIFIER_PATH, OUTPUT_FOLDER,
    NUM_TIME, NUM_FEAT, NUM_AUX, NUM_GROUPS, DATASET,
    LONGITUDINAL_COST_VECTOR, STATIC_COST_VECTOR,
)
from dataset import (
    load_synthetic_data, load_cheears_data, load_cheears_day_context_data,
    load_klg_data, load_womac_data,
    load_ILIADD_data, load_adni_data,
)
from models import Predictor, MaskLayer
from utils import set_seed, build_group_to_feat_matrix

CSV_PATH = 'vanilla.csv'
METHOD_NAME = 'all context'


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _feat_to_group_mask(feat_mask_np, group_to_feat_matrix, num_time, num_feat, num_groups):
    """Convert feature-level availability mask to group-level mask.

    Args:
        feat_mask_np: (B, T*num_feat) numpy array
        group_to_feat_matrix: (num_groups, num_feat) torch tensor
        Returns (B, T*num_groups) numpy array
    """
    B = feat_mask_np.shape[0]
    f = torch.from_numpy(feat_mask_np).float().reshape(B, num_time, num_feat)
    g = torch.matmul(f, group_to_feat_matrix.T)   # (B, T, num_groups)
    g = (g > 0).float()
    return g.reshape(B, num_time * num_groups).numpy()


def evaluate_vanilla(csv_path=CSV_PATH):
    set_seed(42)

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier not found at {CLASSIFIER_PATH}. "
            f"Please run train_classifier.py first."
        )

    # ── load classifier ────────────────────────────────────────────────
    ckpt = torch.load(CLASSIFIER_PATH, map_location='cpu')
    num_time = ckpt['num_time']
    num_feat = ckpt['num_feat']
    num_aux  = ckpt.get('num_aux', 0)
    y_dim    = ckpt['y_dim']

    predictor = Predictor(
        d_in=num_time * num_feat + num_aux,
        d_out=y_dim,
        hidden=ckpt['config']['hidden_dim'],
        dropout=ckpt['config']['dropout'],
    )
    predictor.load_state_dict(ckpt['predictor'])

    mask_layer = MaskLayer(mask_size=num_time * num_feat, append=False)
    mask_layer.load_state_dict(ckpt['mask_layer'])

    device = _get_device()
    predictor.eval().to(device)
    mask_layer.eval().to(device)

    # ── group-to-feat matrix (cheears only) ───────────────────────────
    num_groups = NUM_GROUPS
    use_groups = (num_groups != num_feat)
    group_to_feat_matrix = None
    if use_groups:
        group_to_feat_matrix = build_group_to_feat_matrix(num_feat=num_feat)

    fc = np.array(LONGITUDINAL_COST_VECTOR, dtype=np.float32)   # (num_groups,)
    feature_costs_flat = np.tile(fc, num_time)                   # (T*num_groups,)
    afc = np.array(STATIC_COST_VECTOR, dtype=np.float32)         # (num_aux,)

    # ── load test data ─────────────────────────────────────────────────
    test_path = os.path.join(DATA_FOLDER, 'test_data.npz')
    if DATASET in ('cheears', 'cheears_demog', 'cheears_day_context'):
        loader = load_cheears_day_context_data if DATASET == 'cheears_day_context' else load_cheears_data
        test_ds = loader(test_path)
    elif DATASET == 'klg':
        test_ds = load_klg_data(test_path)
    elif DATASET == 'womac':
        test_ds = load_womac_data(test_path)
    elif DATASET == 'ILIADD':
        test_ds = load_ILIADD_data(test_path)
    elif DATASET == 'adni':
        test_ds = load_adni_data(test_path)
    else:
        test_ds = load_synthetic_data(test_path)

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ── evaluate ───────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []
    total_long_cost = 0.0
    total_aux_cost  = 0.0
    total_samples   = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Vanilla [{DATASET}]"):
            if len(batch) == 5:
                x, y, m_avail, x_static, mask_static = batch
                x_static    = torch.nan_to_num(x_static).float().to(device)
                mask_static = mask_static.float().to(device)
            else:
                x, y, m_avail = batch
                x_static    = None
                mask_static = None

            x      = torch.nan_to_num(x).to(device)
            y      = y.to(device)
            m_avail = m_avail.to(device)
            B = x.shape[0]

            if x.dim() == 3:
                x_flat      = x.reshape(B, -1)
                m_avail_flat = m_avail.reshape(B, -1)
            else:
                x_flat      = x
                m_avail_flat = m_avail

            # Aux: acquire all available static features
            aux_acquired = None
            if x_static is not None and num_aux > 0:
                aux_acquired = x_static * mask_static
                total_aux_cost += (
                    mask_static.cpu().numpy() * afc
                ).sum(axis=1).sum()

            # Longitudinal cost: all available features acquired
            m_np = m_avail_flat.cpu().numpy()      # (B, T*num_feat)
            if use_groups:
                m_groups_np = _feat_to_group_mask(
                    m_np, group_to_feat_matrix, num_time, num_feat, num_groups
                )
            else:
                m_groups_np = m_np                 # group == feature
            total_long_cost += (m_groups_np * feature_costs_flat).sum(axis=1).sum()

            # Predict at every timestep with all available features up to t
            preds_per_t = []
            for t in range(num_time):
                m_t = m_avail_flat.clone()
                m_t[:, (t + 1) * num_feat:] = 0   # mask future timesteps

                x_t = mask_layer(x_flat, m_t)
                t_ind = torch.full((B,), (t + 1) / num_time, device=device).unsqueeze(1)

                if aux_acquired is not None:
                    x_in = torch.cat([t_ind, x_t, aux_acquired], dim=1)
                else:
                    x_in = torch.cat([t_ind, x_t], dim=1)

                preds_per_t.append(predictor(x_in))   # (B, y_dim)

            pred = torch.stack(preds_per_t, dim=1)     # (B, T, y_dim)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            total_samples += B

    all_preds  = np.concatenate(all_preds,  axis=0)   # (N, T, y_dim)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, T)

    all_preds_flat  = all_preds.reshape(-1, y_dim)
    all_labels_flat = all_labels.reshape(-1)

    valid = all_labels_flat != -1
    all_preds_flat  = all_preds_flat[valid]
    all_labels_flat = all_labels_flat[valid]

    accuracy = (np.argmax(all_preds_flat, axis=-1) == all_labels_flat).mean()

    preds_t  = torch.from_numpy(all_preds_flat).float()
    labels_t = torch.from_numpy(all_labels_flat).long()

    if y_dim == 2:
        auroc = AUROC(task='binary')(preds_t[:, 1], labels_t).item()
        auprc = AveragePrecision(task='binary')(preds_t[:, 1], labels_t).item()
    else:
        auroc = AUROC(task='multiclass', num_classes=y_dim)(preds_t, labels_t).item()
        auprc = AveragePrecision(task='multiclass', num_classes=y_dim)(preds_t, labels_t).item()

    avg_long_cost = total_long_cost / total_samples
    avg_aux_cost  = total_aux_cost  / total_samples
    avg_total_cost = avg_long_cost + avg_aux_cost

    results = {
        'accuracy':      accuracy,
        'auroc':         auroc,
        'auprc':         auprc,
        'avg_cost':      avg_total_cost,
        'avg_long_cost': avg_long_cost,
        'avg_aux_cost':  avg_aux_cost,
    }

    print(f"\n{'='*60}")
    print(f"VANILLA RESULTS — {DATASET}")
    print(f"{'='*60}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUROC    : {auroc:.4f}")
    print(f"AUPRC    : {auprc:.4f}")
    print(f"Total cost (avg): {avg_total_cost:.4f}")
    print(f"  Long cost     : {avg_long_cost:.4f}")
    print(f"  Aux  cost     : {avg_aux_cost:.4f}")
    print(f"{'='*60}\n")

    # ── save to CSV ────────────────────────────────────────────────────
    row = [
        METHOD_NAME, DATASET, '',
        results['accuracy'], results['auroc'], results['auprc'],
        results['avg_cost'], results['avg_long_cost'], results['avg_aux_cost'],
        '', '',   # cw, acw — not applicable
    ]

    file_exists = os.path.exists(csv_path)
    if file_exists and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'rb') as f:
            f.seek(-1, 2)
            if f.read(1) != b'\n':
                with open(csv_path, 'a') as fa:
                    fa.write('\n')

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['method', 'data', '', 'ACC', 'AUROC', 'AUPRC',
                             'total_cost', 'long_cost', 'aux_cost', 'cw', 'acw'])
        writer.writerow(row)

    print(f"Results appended to {csv_path}")


if __name__ == '__main__':
    evaluate_vanilla()
