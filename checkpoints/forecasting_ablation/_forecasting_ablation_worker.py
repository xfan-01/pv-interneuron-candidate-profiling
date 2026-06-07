import sys
from pathlib import Path as _WorkerPath
sys.path.insert(0, str(_WorkerPath.cwd()))

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import gc
import json
import random
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd
import torch
from model.utils.device import get_device
from model.utils.reproducibility import seed_everything

# ---------------------------------------------------------------------
# Core experiment controls
# ---------------------------------------------------------------------
DATASET = 'labelled_data_2'
RUN_SEEDS = [42, 66, 88]  # 3 seeds if possible; reduce to [42] for a smoke test.
CUDA_DEVICE_IDS = [int(x) for x in os.environ.get('CUDA_DEVICE_IDS', '0,1').split(',') if x.strip()]
CUDA_DEVICE_INDEX = CUDA_DEVICE_IDS[0] if CUDA_DEVICE_IDS else 0
MAX_EPOCHS = 100
WARMUP_EPOCHS = 5
PATIENCE = 8
SELECTION_METRIC = 'val_loss'

# One model runs on one GPU. BATCH_SIZE is per model/per GPU.
BATCH_SIZE = 128
ACCUMULATION_STEPS = 1
USE_AMP = True
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
EVAL_BATCH_SIZE = 128
EVAL_SAMPLE_N_FOR_STRUCTURE = 2000

_DATASET_CONFIGS = {
    'labelled_data_2': {
        'data_path': 'data/labelled_data_2.h5ad',
        'n_genes': 2134,
        'max_len': 1000,
        'save_dir': 'checkpoints/forecasting_ablation',
    },
}

assert DATASET in _DATASET_CONFIGS, f"Unknown DATASET '{DATASET}'."
_ds = _DATASET_CONFIGS[DATASET]


def resolve_cuda_devices(device_ids):
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        bad_ids = [idx for idx in device_ids if idx >= n_devices]
        if bad_ids:
            raise ValueError(f'CUDA_DEVICE_IDS={device_ids} but only {n_devices} CUDA device(s) are visible.')
        torch.cuda.set_device(device_ids[0])
        return torch.device(f'cuda:{device_ids[0]}'), device_ids
    return get_device(0), []


DEVICE, CUDA_DEVICE_IDS = resolve_cuda_devices(CUDA_DEVICE_IDS)
USE_DATA_PARALLEL = False

TRAJ_CONFIG = {
    'branch_map': {'hGPC': 0, 'PV': 1, 'NPV': 2},
    'model_params': {
        'n_genes': _ds['n_genes'],
        'max_len': _ds['max_len'],
        'd_model': 256,
        'n_heads': 4,
        'n_encoder_layers': 4,
        'n_query_self_layers': 2,
        'n_query_cross_layers': 1,
        'dropout': 0.1,
        'lr': 1e-4,
        'epochs': MAX_EPOCHS,
        'batch_size': BATCH_SIZE,
        'device': DEVICE,
    },
    'loss_weights': {
        'lambda_nz': 1.0,
        'lambda_z_l1': 1.0,
        'lambda_z_l2': 1.0,
        'lambda_nzcos': 0.5,
        'min_genes_for_cos': 5,
    },
    'mask_prob': 0.30,
    'base_mask_prob': 0.30,
    'reconstruct_self_mask_prob_nz': 0.30,
    'reconstruct_self_mask_prob_z': 0.05,
    'time_jitter_std': 0.005,
    'data_path': _ds['data_path'],
    'save_dir': _ds['save_dir'],
    'use_accumulation': True,
    'accumulation_steps': ACCUMULATION_STEPS,
    'parallel': USE_DATA_PARALLEL,
    'device_ids': CUDA_DEVICE_IDS,
}

ABLATION_VARIANTS = [
    {
        'name': 'full_model',
        'label': 'Full model',
        'use_query_self_attention': True,
        'use_query_cross_attention': True,
        'plain_decoder': False,
    },
    {
        'name': 'wo_query_self_attention',
        'label': 'w/o query self-attention',
        'use_query_self_attention': False,
        'use_query_cross_attention': True,
        'plain_decoder': False,
    },
    {
        'name': 'wo_query_cross_attention',
        'label': 'w/o query cross-attention',
        'use_query_self_attention': True,
        'use_query_cross_attention': False,
        'plain_decoder': False,
    },
    {
        'name': 'plain_decoder',
        'label': 'Plain decoder',
        'use_query_self_attention': False,
        'use_query_cross_attention': False,
        'plain_decoder': True,
    },
]

os.makedirs(TRAJ_CONFIG['save_dir'], exist_ok=True)

print('Dataset:', DATASET)
print('Device:', TRAJ_CONFIG['model_params']['device'])
if torch.cuda.is_available():
    print('CUDA devices:', CUDA_DEVICE_IDS)
    print('Primary CUDA device name:', torch.cuda.get_device_name(TRAJ_CONFIG['model_params']['device']))
print('One model per GPU:', torch.cuda.is_available() and len(CUDA_DEVICE_IDS) > 0)
print('DataParallel:', USE_DATA_PARALLEL)
print('Save dir:', TRAJ_CONFIG['save_dir'])
print('Seeds:', RUN_SEEDS)
print('Max epochs:', MAX_EPOCHS, '| Warmup epochs:', WARMUP_EPOCHS, '| Patience:', PATIENCE)
print('Batch size:', BATCH_SIZE, '| Accumulation:', ACCUMULATION_STEPS, '| AMP:', USE_AMP)

import scanpy as sc
import numpy as np
import pandas as pd
import torch
import scipy.sparse
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, i):
        return self.data_list[i]


class PrepareTrajectoryData:
    def __init__(
        self,
        h5ad_path,
        config,
        subset_col='trajectory_class',
        subset_values=('PV',),
        time_col='refined_pseudotime',
        cluster_col='leiden_0.2_c6',
        n_bins=120,
        allowed_offsets=(1, 2, 3, 4, 5, 6),   # 允许的 Horizon 跨度
        base_max_dist=12.0,                   # 基础距离阈值 (h=1)
        dist_alpha=1.0,                       # 动态距离松弛系数
        allowed_cross_steps=(1, 2),           # 允许的跨 Cluster rank 步数
        k_intra=1,                            # A类(同cluster)最多保留1个
        k_cross=2,                            # B类(跨cluster)最多保留2个
        val_split=0.2,                        # 每个 bin-pair 内部隔离 20% 源细胞
        heldout_split=0.1,                    # 额外隔离 10% 源细胞作为 held-out
        pair_diagnostics=True,
    ):
        self.adata = sc.read_h5ad(h5ad_path)
        self.cfg = config['model_params']
        self.max_len = self.cfg['max_len']
        self.n_genes = self.cfg['n_genes']

        # -------------------------
        # 1. 基础过滤与时间归一化 (与之前一致)
        # -------------------------
        if subset_col is not None and subset_values is not None:
            self.adata = self.adata[self.adata.obs[subset_col].isin(subset_values)].copy()

        if cluster_col not in self.adata.obs.columns:
            leiden_cols = [c for c in self.adata.obs.columns if str(c).startswith('leiden')]
            cluster_col = leiden_cols[0] if leiden_cols else subset_col

        times = pd.to_numeric(self.adata.obs[time_col], errors='coerce').to_numpy(dtype=np.float32)
        valid_time_mask = ~np.isnan(times)
        self.adata = self.adata[valid_time_mask].copy()
        times = times[valid_time_mask]
        
        t_min, t_max = np.min(times), np.max(times)
        norm_times = (times - t_min) / (t_max - t_min + 1e-8) if not np.isclose(t_min, t_max) else np.zeros_like(times)
        self.adata.obs['norm_time'] = norm_times

        raw_data = self.adata.X.toarray() if scipy.sparse.issparse(self.adata.X) else np.asarray(self.adata.X)
        raw_data = np.asarray(raw_data, dtype=np.float32)

        if raw_data.shape[1] < self.n_genes:
            self.n_genes = raw_data.shape[1]

        # -------------------------
        # 2. PCA 坐标提取 (修复 Sparse Warning 的注释与逻辑)
        # -------------------------
        if 'X_pca' in self.adata.obsm and self.adata.obsm['X_pca'] is not None:
            self.coords = np.asarray(self.adata.obsm['X_pca'], dtype=np.float32).copy()
        else:
            adata_for_pca = self.adata.copy()
            try:
                sc.pp.highly_variable_genes(adata_for_pca, min_mean=0.0125, max_mean=3, min_disp=0.5)
                if 'highly_variable' in adata_for_pca.var.columns and adata_for_pca.var['highly_variable'].sum() >= 2:
                    adata_for_pca = adata_for_pca[:, adata_for_pca.var['highly_variable']].copy()
            except Exception:
                pass

            # Note: Dense conversion only occurs in the PCA fallback branch, preferably after HVG filtering.
            if scipy.sparse.issparse(adata_for_pca.X):
                adata_for_pca.X = adata_for_pca.X.toarray()
                
            sc.pp.scale(adata_for_pca, max_value=10)
            n_comps = min(50, min(adata_for_pca.n_obs, adata_for_pca.n_vars) - 1)

            try:
                sc.tl.pca(adata_for_pca, svd_solver='arpack', n_comps=n_comps)
                self.coords = np.asarray(adata_for_pca.obsm['X_pca'], dtype=np.float32).copy()
            except:
                self.coords = np.asarray(adata_for_pca.X, dtype=np.float32)
            del adata_for_pca

        coord_mean = self.coords.mean(axis=0, keepdims=True)
        coord_std = self.coords.std(axis=0, keepdims=True)
        coord_std[coord_std < 1e-8] = 1.0
        self.coords = (self.coords - coord_mean) / coord_std
        self.coords = np.asarray(self.coords, dtype=np.float32)

        # -------------------------
        # 3. Cluster Rank & Time Bins
        # -------------------------
        obs_df = self.adata.obs.copy()
        cluster_mean_times = obs_df.groupby(cluster_col, observed=True)[time_col].mean().sort_values()
        cluster_rank_dict = {c: rank for rank, c in enumerate(cluster_mean_times.index)}
        cell_ranks = np.array([cluster_rank_dict[c] for c in obs_df[cluster_col]], dtype=np.int32)

        obs_df['time_bin'] = pd.qcut(obs_df[time_col], q=n_bins, labels=False, duplicates='drop')
        valid_bin_mask = ~obs_df['time_bin'].isna().to_numpy()
        
        self.adata = self.adata[valid_bin_mask].copy()
        time_bin_np = obs_df.loc[valid_bin_mask, 'time_bin'].astype(int).to_numpy()
        self.adata.obs['time_bin'] = time_bin_np
        raw_data = raw_data[valid_bin_mask]
        norm_times = norm_times[valid_bin_mask]
        self.coords = self.coords[valid_bin_mask]
        cell_ranks = cell_ranks[valid_bin_mask]
        
        actual_bins = np.sort(np.unique(time_bin_np))

        # -------------------------
        # 4. NEW: Per-Bin-Pair Source-Disjoint + Multi-Horizon Construction
        # -------------------------
        self.train_data = []
        self.val_data = []
        self.heldout_data = []
        
        # Diagnostics trackers
        intra_count = 0
        cross_count = 0
        horizon_stats = {offset: 0 for offset in allowed_offsets}
        cluster_source_counts = {c: 0 for c in cluster_mean_times.index}

        rng = np.random.default_rng(42)

        # 外层循环：遍历每一个 source bin
        for i, curr_bin in enumerate(actual_bins):
            idx_curr = np.where(time_bin_np == curr_bin)[0]
            if len(idx_curr) == 0:
                continue

            # 内层循环：遍历允许的 horizon
            for offset in allowed_offsets:
                target_bin_idx = i + offset
                if target_bin_idx >= len(actual_bins):
                    continue
                    
                next_bin = actual_bins[target_bin_idx]
                idx_next = np.where(time_bin_np == next_bin)[0]
                if len(idx_next) == 0:
                    continue

                # 必改3：引入 Dynamic max_dist
                current_max_dist = base_max_dist + dist_alpha * (offset - 1)
                
                # 用于收集当前 (curr_bin, next_bin) 下的所有合法候选对
                # 结构: {source_idx: [ ('intra', item), ('cross', item) ]}
                transition_candidates = {}

                for c_idx in idx_curr:
                    curr_coord = self.coords[c_idx].reshape(1, -1)
                    next_coords = self.coords[idx_next]
                    dists = np.linalg.norm(next_coords - curr_coord, axis=1)

                    safe_mask = dists <= current_max_dist
                    valid_idx_next = idx_next[safe_mask]
                    valid_dists = dists[safe_mask]

                    if len(valid_idx_next) == 0:
                        continue

                    rank_s = cell_ranks[c_idx]
                    rank_t = cell_ranks[valid_idx_next]

                    # 必改4：放宽 cross-cluster 限制
                    mask_intra = (rank_t == rank_s)
                    mask_cross = np.isin(rank_t - rank_s, allowed_cross_steps)

                    source_pairs = []

                    # A类：Intra-cluster
                    idx_intra = valid_idx_next[mask_intra]
                    dist_intra = valid_dists[mask_intra]
                    if len(idx_intra) > 0:
                        k_i = min(k_intra, len(idx_intra))
                        top_intra = idx_intra[np.argsort(dist_intra)[:k_i]]
                        for t_idx in top_intra:
                            source_pairs.append(('intra', self._create_item(raw_data, norm_times, c_idx, t_idx)))

                    # B类：Boundary-crossing
                    idx_cross = valid_idx_next[mask_cross]
                    dist_cross = valid_dists[mask_cross]
                    if len(idx_cross) > 0:
                        k_c = min(k_cross, len(idx_cross))
                        top_cross = idx_cross[np.argsort(dist_cross)[:k_c]]
                        for t_idx in top_cross:
                            source_pairs.append(('cross', self._create_item(raw_data, norm_times, c_idx, t_idx)))

                    if source_pairs:
                        transition_candidates[c_idx] = source_pairs

                # 必改1 & 5：在具体的 bin-pair 级别进行 source-disjoint split
                if not transition_candidates:
                    continue
                    
                unique_sources_in_transition = list(transition_candidates.keys())
                
                # 记录 Source cluster diagnostics
                for s_idx in unique_sources_in_transition:
                    cluster_name = obs_df[cluster_col].iloc[s_idx]
                    cluster_source_counts[cluster_name] += 1

                rng.shuffle(unique_sources_in_transition)
                n_sources = len(unique_sources_in_transition)

                if n_sources >= 3:
                    n_heldout = int(np.floor(heldout_split * n_sources))
                    n_heldout = min(max(n_heldout, 1), n_sources - 2)
                else:
                    n_heldout = 0

                remaining_sources = unique_sources_in_transition[n_heldout:]
                n_remaining = len(remaining_sources)

                if n_remaining >= 2:
                    n_val = int(np.floor(val_split * n_remaining))
                    n_val = min(max(n_val, 1), n_remaining - 1)
                else:
                    n_val = 0

                heldout_sources = set(unique_sources_in_transition[:n_heldout])
                val_sources = set(remaining_sources[:n_val])

                for c_idx, pairs in transition_candidates.items():
                    if c_idx in heldout_sources:
                        target_list = self.heldout_data
                    elif c_idx in val_sources:
                        target_list = self.val_data
                    else:
                        target_list = self.train_data
                    
                    for p_type, item in pairs:
                        target_list.append(item)
                        # 更新全局 Diagnostics
                        horizon_stats[offset] += 1
                        if p_type == 'intra': 
                            intra_count += 1
                        else: 
                            cross_count += 1

        if pair_diagnostics:
            total_pairs = len(self.train_data) + len(self.val_data) + len(self.heldout_data)
            print("="*50)
            print(f"Dataset Pipeline Finished (Multi-Horizon & Per-Bin-Pair Split)")
            print(f"Total pairs: {total_pairs} (Train: {len(self.train_data)}, Val: {len(self.val_data)}, Held-out: {len(self.heldout_data)})")
            print(f"A Class (Intra-cluster) pairs: {intra_count}")
            print(f"B Class (Boundary-cross) pairs: {cross_count}")
            print("\nPairs per Horizon (Offset):")
            for offset in allowed_offsets:
                print(f"  Offset {offset}: {horizon_stats[offset]} pairs")
            print("\nTop 5 Source Clusters by Contribution:")
            sorted_clusters = sorted(cluster_source_counts.items(), key=lambda x: x[1], reverse=True)
            for c_name, count in sorted_clusters[:5]:
                print(f"  {c_name}: {count} unique source participations")
            print("="*50)

    def _create_item(self, data, times, i, j):
        gene_id, gene_val, padding_mask = self._preprocess_numpy(data[i])
        return {
            'gene_id': gene_id,
            'gene_val': gene_val,
            'padding_mask': padding_mask,
            'full_input_val': torch.tensor(data[i][:self.n_genes], dtype=torch.float32),
            'time': torch.tensor(times[i], dtype=torch.float32),
            'target_time': torch.tensor(times[j], dtype=torch.float32),
            'target_val': torch.tensor(data[j][:self.n_genes], dtype=torch.float32),
            'c_idx': i,
            'target_idx': j
        }
    def _preprocess_numpy(self, expr):
        expr = np.asarray(expr, dtype=np.float32)
        nz = np.where(expr > 0)[0]
        nz = nz[nz < self.n_genes]

        if len(nz) == 0:
            return (
                torch.zeros(self.max_len, dtype=torch.long),
                torch.zeros(self.max_len, dtype=torch.float32),
                torch.ones(self.max_len, dtype=torch.bool)
            )

        idx = nz + 1 
        val = expr[nz]
        L = self.max_len

        if len(idx) > L:
            idx = idx[:L]
            val = val[:L]
            pad_mask = np.zeros(L, dtype=bool)
        else:
            pad_len = L - len(idx)
            idx = np.pad(idx, (0, pad_len), constant_values=0)
            val = np.pad(val, (0, pad_len), constant_values=0.0)
            pad_mask = np.pad(np.zeros(len(nz), dtype=bool), (0, pad_len), constant_values=True)

        return (
            torch.tensor(idx, dtype=torch.long),
            torch.tensor(val, dtype=torch.float32),
            torch.tensor(pad_mask, dtype=torch.bool)
        )

from torch.utils.data import DataLoader

# Build the split once. Every ablation and seed uses these same lists.
seed_everything(42)
processor = PrepareTrajectoryData(
    h5ad_path=TRAJ_CONFIG['data_path'],
    config=TRAJ_CONFIG,
    subset_col='trajectory_class',
    subset_values=('PV',),
    time_col='refined_pseudotime',
    cluster_col='leiden_0.2_c6',
    n_bins=120,
    base_max_dist=12.0,
    dist_alpha=1.0,
    allowed_offsets=(1, 2, 3, 4, 5, 6),
    allowed_cross_steps=(1, 2),
    k_intra=1,
    k_cross=2,
    val_split=0.2,
    heldout_split=0.1,
    pair_diagnostics=True,
)

train_set = TrajectoryDataset(processor.train_data)
val_set = TrajectoryDataset(processor.val_data)
heldout_set = TrajectoryDataset(processor.heldout_data)

def make_loader(dataset, batch_size, shuffle, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

print(f'Dataset size: {len(train_set) + len(val_set) + len(heldout_set)}')
print(f'Train size: {len(train_set)}')
print(f'Validation size: {len(val_set)}')
print(f'Held-out size: {len(heldout_set)}')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneEncoder(nn.Module):
    def __init__(self, n_genes, d_model):
        super().__init__()
        self.n_genes = n_genes
        self.gene_emb = nn.Embedding(n_genes + 1, d_model, padding_idx=0)
        self.val_proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, g_id, g_val):
        # 0 保留给 padding；合法 gene id 应在 [0, n_genes] 内
        if torch.any(g_id < 0) or torch.any(g_id > self.n_genes):
            bad_min = g_id.min().item()
            bad_max = g_id.max().item()
            raise ValueError(
                f"g_id out of range. Expected ids in [0, {self.n_genes}], "
                f"but got min={bad_min}, max={bad_max}."
            )

        return self.norm(self.gene_emb(g_id) + self.val_proj(g_val.unsqueeze(-1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t):
        t_scaled = t * 100.0
        half_dim = self.d_model // 2
        freq = math.log(10000.0) / max(1, (half_dim - 1))
        freq = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32) * -freq
        )

        emb = t_scaled.unsqueeze(-1).float() * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb).unsqueeze(1)  # [B, 1, D]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, src, key_padding_mask=None, need_weights=True):
        attn_out, attn_weights = self.self_attn(
            src,
            src,
            src,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        src = self.norm1(src + self.dropout1(attn_out))
        src = self.norm2(src + self.dropout2(self.ffn(src)))
        return src, attn_weights


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.w3(F.silu(self.w2(x)) * self.w1(x))


class QuerySelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = SwiGLU(d_model, int(d_model * 8 / 3))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt):
        x, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        x = self.norm1(tgt + self.dropout(x))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class QueryCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = SwiGLU(d_model, int(d_model * 8 / 3))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_key_padding_mask=None, need_weights=False):
        y, cross_attn_weights = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        x = self.norm1(tgt + self.dropout(y))
        out = self.norm2(x + self.dropout(self.feed_forward(x)))
        if need_weights:
            return out, cross_attn_weights
        return out


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["model_params"]

        n_encoder_layers = int(cfg.get("n_encoder_layers", cfg.get("n_layers", 2)))
        n_query_self_layers = int(cfg.get("n_query_self_layers", 2))
        n_query_cross_layers = int(cfg.get("n_query_cross_layers", 1))

        self.d_model = cfg["d_model"]
        self.n_genes = cfg["n_genes"]

        self.gene_encoder = GeneEncoder(self.n_genes, self.d_model)

        # 三种时间：source_time, target_time, delta_time
        self.source_time_encoder = SinusoidalTimeEmbedding(self.d_model)
        self.target_time_encoder = SinusoidalTimeEmbedding(self.d_model)
        self.delta_time_encoder = SinusoidalTimeEmbedding(self.d_model)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(self.d_model, cfg["n_heads"], cfg["dropout"])
                for _ in range(n_encoder_layers)
            ]
        )

        self.gene_queries = nn.Parameter(torch.randn(self.n_genes, self.d_model))

        self.query_self_layers = nn.ModuleList(
            [
                QuerySelfAttentionBlock(self.d_model, cfg["n_heads"], cfg["dropout"])
                for _ in range(n_query_self_layers)
            ]
        )

        self.query_cross_layers = nn.ModuleList(
            [
                QueryCrossAttentionBlock(self.d_model, cfg["n_heads"], cfg["dropout"])
                for _ in range(n_query_cross_layers)
            ]
        )

        # 把 delta_time 的条件也明确注入到最终预测头前
        self.pred_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, 1),
        )

    def forward(
        self,
        g_id,
        g_val,
        source_time,
        target_time,
        padding_mask=None,
        need_weights=False,
        epoch=None,
    ):
        # delta_time 明确建模
        delta_time = target_time - source_time

        # encoder: source cell + source_time + delta_time
        src_t_emb = self.source_time_encoder(source_time)   # [B, 1, D]
        delta_t_emb = self.delta_time_encoder(delta_time)   # [B, 1, D]

        x = self.gene_encoder(g_id, g_val) + src_t_emb + delta_t_emb

        enc_attn_weights = None
        for layer in self.encoder_layers:
            x, enc_attn_weights = layer(
                x,
                key_padding_mask=padding_mask,
                need_weights=need_weights,
            )

        # decoder queries: gene query + target_time + delta_time
        bsz = x.size(0)
        queries = self.gene_queries.unsqueeze(0).expand(bsz, -1, -1)

        tgt_t_emb = self.target_time_encoder(target_time)   # [B, 1, D]
        out = queries + tgt_t_emb + delta_t_emb

        # 如果 n_query_self_layers = 0，这里会自动跳过
        for layer in self.query_self_layers:
            out = layer(out)

        cross_attn_weights = None
        for layer in self.query_cross_layers:
            if need_weights:
                out, cross_attn_weights = layer(
                    out,
                    memory=x,
                    memory_key_padding_mask=padding_mask,
                    need_weights=True,
                )
            else:
                out = layer(
                    out,
                    memory=x,
                    memory_key_padding_mask=padding_mask,
                    need_weights=False,
                )

        raw_pred = self.pred_head(out).squeeze(-1)

        # 用 Softplus 代替 ReLU，保证非负且梯度更平滑
        final_pred = F.softplus(raw_pred)

        if need_weights:
            return torch.clamp(final_pred, min=0.0, max=1e5), enc_attn_weights, cross_attn_weights
        return torch.clamp(final_pred, min=0.0, max=1e5), None, None

class AblationGenerator(nn.Module):
    """Generator with switchable query self-attention, query cross-attention, and plain MLP decoder."""

    def __init__(self, config, variant):
        super().__init__()
        cfg = config['model_params']
        self.variant = dict(variant)
        self.use_query_self_attention = bool(variant.get('use_query_self_attention', True))
        self.use_query_cross_attention = bool(variant.get('use_query_cross_attention', True))
        self.plain_decoder = bool(variant.get('plain_decoder', False))

        n_encoder_layers = int(cfg.get('n_encoder_layers', cfg.get('n_layers', 2)))
        n_query_self_layers = int(cfg.get('n_query_self_layers', 2))
        n_query_cross_layers = int(cfg.get('n_query_cross_layers', 1))

        self.d_model = cfg['d_model']
        self.n_genes = cfg['n_genes']
        self.gene_encoder = GeneEncoder(self.n_genes, self.d_model)
        self.source_time_encoder = SinusoidalTimeEmbedding(self.d_model)
        self.target_time_encoder = SinusoidalTimeEmbedding(self.d_model)
        self.delta_time_encoder = SinusoidalTimeEmbedding(self.d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerBlock(self.d_model, cfg['n_heads'], cfg['dropout'])
            for _ in range(n_encoder_layers)
        ])

        if self.plain_decoder:
            # Plain decoder avoids the [B, n_genes, d_model] query attention stack.
            self.plain_decoder_head = nn.Sequential(
                nn.Linear(self.d_model * 3, self.d_model * 2),
                nn.SiLU(),
                nn.Dropout(cfg['dropout']),
                nn.Linear(self.d_model * 2, self.n_genes),
            )
        else:
            self.gene_queries = nn.Parameter(torch.randn(self.n_genes, self.d_model))
            self.query_self_layers = nn.ModuleList([
                QuerySelfAttentionBlock(self.d_model, cfg['n_heads'], cfg['dropout'])
                for _ in range(n_query_self_layers if self.use_query_self_attention else 0)
            ])
            self.query_cross_layers = nn.ModuleList([
                QueryCrossAttentionBlock(self.d_model, cfg['n_heads'], cfg['dropout'])
                for _ in range(n_query_cross_layers if self.use_query_cross_attention else 0)
            ])
            self.pred_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.SiLU(),
                nn.Linear(self.d_model // 2, 1),
            )

    @staticmethod
    def masked_mean(x, padding_mask):
        if padding_mask is None:
            return x.mean(dim=1)
        valid = (~padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / denom

    def forward(self, g_id, g_val, source_time, target_time, padding_mask=None, need_weights=False, epoch=None):
        delta_time = target_time - source_time
        src_t_emb = self.source_time_encoder(source_time)
        tgt_t_emb = self.target_time_encoder(target_time)
        delta_t_emb = self.delta_time_encoder(delta_time)

        x = self.gene_encoder(g_id, g_val) + src_t_emb + delta_t_emb

        enc_attn_weights = None
        for layer in self.encoder_layers:
            x, enc_attn_weights = layer(x, key_padding_mask=padding_mask, need_weights=need_weights)

        if self.plain_decoder:
            pooled = self.masked_mean(x, padding_mask)
            cond = torch.cat([pooled, tgt_t_emb.squeeze(1), delta_t_emb.squeeze(1)], dim=-1)
            raw_pred = self.plain_decoder_head(cond)
            final_pred = F.softplus(raw_pred)
            return torch.clamp(final_pred, min=0.0, max=1e5), enc_attn_weights if need_weights else None, None

        bsz = x.size(0)
        queries = self.gene_queries.unsqueeze(0).expand(bsz, -1, -1)
        out = queries + tgt_t_emb + delta_t_emb

        for layer in self.query_self_layers:
            out = layer(out)

        cross_attn_weights = None
        for layer in self.query_cross_layers:
            if need_weights:
                out, cross_attn_weights = layer(out, memory=x, memory_key_padding_mask=padding_mask, need_weights=True)
            else:
                out = layer(out, memory=x, memory_key_padding_mask=padding_mask, need_weights=False)

        raw_pred = self.pred_head(out).squeeze(-1)
        final_pred = F.softplus(raw_pred)
        if need_weights:
            return torch.clamp(final_pred, min=0.0, max=1e5), enc_attn_weights, cross_attn_weights
        return torch.clamp(final_pred, min=0.0, max=1e5), None, None


import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualHybridLoss(nn.Module):
    def __init__(
        self,
        lambda_nz=1.0,
        lambda_z_l1=0.2,
        lambda_z_l2=0.2,
        lambda_nzcos=0.2,
        min_genes_for_cos=5,
        **kwargs
    ):
        super().__init__()
        self.lambda_nz = float(lambda_nz)
        self.lambda_z_l1 = float(lambda_z_l1)
        self.lambda_z_l2 = float(lambda_z_l2)
        self.lambda_nzcos = float(lambda_nzcos)
        self.min_genes_for_cos = int(min_genes_for_cos)

    def forward(self, pred, target, active_mask=None, dummy_gate=None):
        """
        pred:   [B, G]
        target: [B, G]
        active_mask: [B, G] bool/float, only positions with active_mask==1 participate in loss
        """
        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite prediction detected in loss input `pred`.")

        pred = pred.float()
        target = target.float()

        if active_mask is None:
            active_mask = torch.ones_like(target, dtype=pred.dtype, device=pred.device)
        else:
            active_mask = active_mask.float()

        # biological masks
        nz_mask = (target > 0).float() * active_mask
        z_mask = (target <= 0).float() * active_mask

        # -------------------------
        # 1) non-zero SmoothL1 loss
        # -------------------------
        smooth_l1 = F.smooth_l1_loss(pred, target, reduction='none')
        num_nz_per_cell = nz_mask.sum(dim=1).clamp(min=1.0)
        loss_nz_per_cell = (smooth_l1 * nz_mask).sum(dim=1) / num_nz_per_cell

        valid_nz_cells = (nz_mask.sum(dim=1) > 0)
        if valid_nz_cells.any():
            loss_nz = loss_nz_per_cell[valid_nz_cells].mean()
        else:
            loss_nz = pred.new_tensor(0.0)

        # -------------------------
        # 2) zero-region MSE loss
        # -------------------------
        diff_sq = (pred - target).pow(2).clamp(max=1e4)
        num_z_per_cell = z_mask.sum(dim=1).clamp(min=1.0)
        loss_z_per_cell = (diff_sq * z_mask).sum(dim=1) / num_z_per_cell

        valid_z_cells = (z_mask.sum(dim=1) > 0)
        if valid_z_cells.any():
            loss_z = loss_z_per_cell[valid_z_cells].mean()
        else:
            loss_z = pred.new_tensor(0.0)

        # -------------------------
        # 3) zero-region L1 loss
        # -------------------------
        loss_z_l1_per_cell = (pred.abs() * z_mask).sum(dim=1) / num_z_per_cell
        if valid_z_cells.any():
            loss_z_l1 = loss_z_l1_per_cell[valid_z_cells].mean()
        else:
            loss_z_l1 = pred.new_tensor(0.0)

        # -------------------------
        # 4) non-zero cosine loss
        # -------------------------
        nz_count_per_cell = nz_mask.sum(dim=1)
        valid_cos_cells = nz_count_per_cell >= self.min_genes_for_cos

        if valid_cos_cells.any():
            pred_nz = pred * nz_mask
            target_nz = target * nz_mask
            cos_sim = F.cosine_similarity(
                pred_nz[valid_cos_cells],
                target_nz[valid_cos_cells],
                dim=1,
                eps=1e-8
            )
            loss_nzcos = 1.0 - cos_sim.mean()
        else:
            loss_nzcos = pred.new_tensor(0.0)

        total_loss = (
            self.lambda_nz * loss_nz +
            self.lambda_z_l2 * loss_z +
            self.lambda_z_l1 * loss_z_l1 +
            self.lambda_nzcos * loss_nzcos
        )

        return (
            total_loss,
            loss_nz.item(),
            loss_nzcos.item(),
            loss_z.item(),
            loss_z_l1.item()
        )

import torch.optim as optim
from contextlib import nullcontext
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

MASK_TOKEN_VAL = -1.0

def amp_autocast_ctx(device):
    if USE_AMP and device.type == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def apply_time_jitter(source_time, target_time, noise_std):
    if noise_std <= 0.0:
        return source_time, target_time
    return (
        torch.clamp(source_time + torch.randn_like(source_time) * noise_std, 0.0, 1.0),
        torch.clamp(target_time + torch.randn_like(target_time) * noise_std, 0.0, 1.0),
    )


def build_optimizer_scheduler(model_obj, cfg):
    optimizer = optim.AdamW(model_obj.parameters(), lr=cfg['model_params']['lr'], weight_decay=0.01)
    total_epochs = int(cfg['model_params']['epochs'])
    warmup_epochs = int(WARMUP_EPOCHS)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    return optimizer, scheduler


def build_active_gene_mask_from_token_mask(s_id, token_mask, pred_dim):
    valid_token_mask = token_mask & (s_id > 0) & (s_id <= pred_dim)
    if not valid_token_mask.any().item():
        return None
    token_gene_idx = (s_id - 1).clamp(min=0, max=pred_dim - 1)
    active_gene_scores = torch.zeros(s_id.size(0), pred_dim, device=s_id.device, dtype=torch.float32)
    active_gene_scores.scatter_add_(1, token_gene_idx, valid_token_mask.float())
    return active_gene_scores > 0


def sample_mask_and_corrupt_values(s_val, padding_mask, mask_prob):
    random_mask = torch.rand(s_val.shape, device=s_val.device) < mask_prob
    valid_genes_mask = ~padding_mask
    actual_mask = random_mask & valid_genes_mask

    s_val_masked = s_val.clone()
    rand_selector = torch.rand(s_val.shape, device=s_val.device)
    mask_replace = actual_mask & (rand_selector < 0.8)
    mask_random = actual_mask & (rand_selector >= 0.8) & (rand_selector < 0.9)
    s_val_masked[mask_replace] = MASK_TOKEN_VAL
    s_val_masked[mask_random] = torch.rand_like(s_val_masked[mask_random]) * 5.0
    supervised_token_mask = mask_replace | mask_random
    return s_val_masked, supervised_token_mask


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, device=None, seed_epoch=0, train=True):
    model.train(train)
    totals = np.zeros(6, dtype=np.float64)
    mask_prob = float(TRAJ_CONFIG.get('base_mask_prob', TRAJ_CONFIG.get('mask_prob', 0.30)))
    time_jitter_std = float(TRAJ_CONFIG.get('time_jitter_std', 0.0)) if train else 0.0
    accumulation_steps = max(1, int(TRAJ_CONFIG.get('accumulation_steps', 1))) if train else 1

    if train:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            s_id = batch['gene_id'].to(device, non_blocking=True)
            s_val = batch['gene_val'].to(device, non_blocking=True)
            mask = batch['padding_mask'].to(device, non_blocking=True)
            s_time = batch['time'].to(device, non_blocking=True)
            target_time = batch['target_time'].to(device, non_blocking=True)
            t_val = batch['target_val'].to(device, non_blocking=True)

            s_time_in, target_time_in = apply_time_jitter(s_time, target_time, time_jitter_std)
            s_val_masked, supervised_token_mask = sample_mask_and_corrupt_values(s_val, mask, mask_prob)

            with amp_autocast_ctx(device):
                preds, _, _ = model(s_id, s_val_masked, s_time_in, target_time_in, padding_mask=mask)
                preds = torch.clamp(preds, min=0.0, max=50.0)
                active_gene_mask = build_active_gene_mask_from_token_mask(s_id, supervised_token_mask, preds.size(1))
                loss_total, nz_smooth_l1, nzcosloss, z_mse, z_l1 = criterion(preds, t_val, active_mask=active_gene_mask)
                loss = loss_total / accumulation_steps

            if train:
                scaler.scale(loss).backward()
                should_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(loader))
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            bs = s_id.size(0)
            totals += np.array([loss_total.item(), nz_smooth_l1, nzcosloss, z_mse, z_l1, bs], dtype=np.float64) * np.array([bs, bs, bs, bs, bs, 1])

            del s_id, s_val, mask, s_time, target_time, t_val, s_time_in, target_time_in
            del s_val_masked, supervised_token_mask, active_gene_mask, preds, loss_total, loss

    denom = max(1.0, totals[5])
    return {
        'loss': totals[0] / denom,
        'nz_smooth_l1': totals[1] / denom,
        'nzcosloss': totals[2] / denom,
        'z_mse': totals[3] / denom,
        'z_l1': totals[4] / denom,
    }


def cleanup_cuda(device=None):
    gc.collect()
    if torch.cuda.is_available():
        if device is None:
            device_indices = range(torch.cuda.device_count())
        else:
            device_indices = [torch.device(device).index]
        for device_idx in device_indices:
            if device_idx is None:
                continue
            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def get_model_predictions(model, dataloader, device):
    model.eval()
    preds_list, trues_list = [], []
    meta = {'c_idx': [], 'target_idx': [], 'time': [], 'target_time': []}

    with torch.inference_mode():
        for batch in dataloader:
            s_id = batch['gene_id'].to(device, non_blocking=True)
            s_val = batch['gene_val'].to(device, non_blocking=True)
            s_time = batch['time'].to(device, non_blocking=True)
            target_time = batch['target_time'].to(device, non_blocking=True)
            mask = batch['padding_mask'].to(device, non_blocking=True)
            with amp_autocast_ctx(device):
                preds, _, _ = model(s_id, s_val, s_time, target_time, padding_mask=mask, need_weights=False)
                preds = torch.clamp(preds, min=0.0, max=50.0)
            preds_list.append(preds.float().cpu().numpy())
            trues_list.append(batch['target_val'].numpy())
            meta['c_idx'].extend(batch['c_idx'].numpy())
            meta['target_idx'].extend(batch['target_idx'].numpy())
            meta['time'].extend(batch['time'].numpy())
            meta['target_time'].extend(batch['target_time'].numpy())
            del s_id, s_val, s_time, target_time, mask, preds
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    preds = np.vstack(preds_list)
    trues = np.vstack(trues_list)
    nz_masks = trues > 0
    return preds, trues, nz_masks, meta


def compute_ablation_metrics(preds, trues, nz_masks, meta, adata, cluster_key='leiden_0.2_c6', sample_n=2000, seed=0):
    target_indices = np.array(meta['target_idx']).astype(int)
    clusters = adata.obs[cluster_key].values[target_indices].astype(str)

    mse = mean_squared_error(trues, preds)
    flat_preds = preds[nz_masks]
    flat_trues = trues[nz_masks]
    global_corr = pearsonr(flat_preds, flat_trues)[0] if flat_trues.size > 1 else np.nan

    true_sparsity = float((trues == 0).mean())
    pred_sparsity = float((preds < 1e-3).mean())
    sparsity_gap = abs(pred_sparsity - true_sparsity)

    if 'time_bin' in adata.obs.columns:
        pair_time_bins = adata.obs['time_bin'].iloc[target_indices].to_numpy()
        valid_bin_mask = ~pd.isna(pair_time_bins)
        valid_bins = pair_time_bins[valid_bin_mask].astype(int)
        valid_preds = preds[valid_bin_mask]
        valid_trues = trues[valid_bin_mask]
        meta_pred_list, meta_true_list = [], []
        for b in np.sort(np.unique(valid_bins)):
            b_mask = valid_bins == b
            if np.any(b_mask):
                meta_pred_list.append(valid_preds[b_mask].mean(axis=0))
                meta_true_list.append(valid_trues[b_mask].mean(axis=0))
        if len(meta_true_list) >= 2:
            meta_preds = np.vstack(meta_pred_list)
            meta_trues = np.vstack(meta_true_list)
            meta_nz = meta_trues > 0
            meta_corr = pearsonr(meta_preds[meta_nz], meta_trues[meta_nz])[0] if meta_trues[meta_nz].size > 1 else np.nan
            n_meta_bins = meta_trues.shape[0]
        else:
            meta_corr = np.nan
            n_meta_bins = len(meta_true_list)
    else:
        meta_corr = np.nan
        n_meta_bins = 0

    n = min(sample_n, preds.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(preds.shape[0], size=n, replace=False) if preds.shape[0] > n else np.arange(preds.shape[0])
    pred_sub = preds[idx]
    cluster_sub = clusters[idx]

    n_components = min(30, pred_sub.shape[0] - 1, pred_sub.shape[1])
    if n_components >= 2 and len(np.unique(cluster_sub)) > 1:
        pred_pca = PCA(n_components=n_components, random_state=seed).fit_transform(pred_sub)
        asw_raw = silhouette_score(pred_pca, cluster_sub)
        asw_scaled = (asw_raw + 1.0) / 2.0
        n_clusters = len(np.unique(cluster_sub))
        cluster_pred = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(pred_pca)
        ari = adjusted_rand_score(cluster_sub, cluster_pred)
        nmi = normalized_mutual_info_score(cluster_sub, cluster_pred)
    else:
        asw_raw = asw_scaled = ari = nmi = np.nan

    return {
        'heldout_mse': mse,
        'global_corr_nonzero': global_corr,
        'meta_cell_corr_by_time_bin': meta_corr,
        'pred_sparsity': pred_sparsity,
        'sparsity_gap': sparsity_gap,
        'true_sparsity': true_sparsity,
        'n_meta_bins': n_meta_bins,
        'asw_cell_type_separation': asw_scaled,
        'asw_raw': asw_raw,
        'ari': ari,
        'nmi': nmi,
        'eval_n': int(n),
    }


def unwrap_model(model_obj):
    return model_obj.module if isinstance(model_obj, torch.nn.DataParallel) else model_obj



def build_ablation_model(variant, device):
    # One worker owns one CUDA device, so the model stays on that GPU only.
    return AblationGenerator(TRAJ_CONFIG, variant).to(device)



def seed_current_worker(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)



def fit_one_run(variant, seed, device_id=None, position=0):
    device = torch.device(f'cuda:{device_id}' if device_id is not None and torch.cuda.is_available() else 'cpu')
    cleanup_cuda(device)
    seed_current_worker(seed, device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True, seed=seed)
    val_loader = make_loader(val_set, EVAL_BATCH_SIZE, shuffle=False)
    heldout_loader = make_loader(heldout_set, EVAL_BATCH_SIZE, shuffle=False)

    model = build_ablation_model(variant, device)
    criterion = ManualHybridLoss(**TRAJ_CONFIG['loss_weights']).to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, TRAJ_CONFIG)
    scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and device.type == 'cuda'))

    run_dir = Path(TRAJ_CONFIG['save_dir']) / variant['name'] / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / 'best_by_val_loss.pth'
    history_path = run_dir / 'history.csv'
    metrics_path = run_dir / 'heldout_metrics.json'

    best_val_loss = float('inf')
    best_epoch = -1
    bad_epochs = 0
    history = []

    gpu_label = f'GPU {device_id}' if device.type == 'cuda' else 'CPU'
    pbar = tqdm(
        range(MAX_EPOCHS),
        desc=f"{gpu_label} | {variant['label']} | seed {seed}",
        unit='ep',
        position=position,
        leave=True,
    )
    for epoch in pbar:
        train_m = run_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler, device=device, train=True, seed_epoch=epoch)
        val_m = run_epoch(model, val_loader, criterion, device=device, train=False, seed_epoch=epoch)
        scheduler.step()

        row = {
            'variant': variant['name'],
            'seed': seed,
            'gpu_id': device_id,
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            **{f'train_{k}': v for k, v in train_m.items()},
            **{f'val_{k}': v for k, v in val_m.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        val_loss = float(val_m['loss'])
        pbar.set_postfix({'train': f"{train_m['loss']:.3f}", 'val': f'{val_loss:.3f}', 'best': f'{best_val_loss:.3f}'})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                'variant': variant,
                'seed': seed,
                'gpu_id': device_id,
                'epoch': epoch,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': best_val_loss,
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best val loss={best_val_loss:.6f}")
                break

    cleanup_cuda(device)
    best_ckpt = torch.load(ckpt_path, map_location='cpu')
    unwrap_model(model).load_state_dict(best_ckpt['model_state_dict'])
    del best_ckpt
    cleanup_cuda(device)
    preds, trues, nz_masks, meta = get_model_predictions(model, heldout_loader, device)
    metrics = compute_ablation_metrics(
        preds, trues, nz_masks, meta, processor.adata,
        cluster_key='leiden_0.2_c6',
        sample_n=EVAL_SAMPLE_N_FOR_STRUCTURE,
        seed=seed,
    )
    metrics.update({
        'variant': variant['name'],
        'label': variant['label'],
        'seed': seed,
        'gpu_id': device_id,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'checkpoint': str(ckpt_path),
    })

    def _make_jsonable(o):
        import numpy as _np
        import torch as _torch
        if isinstance(o, dict):
            return {k: _make_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_make_jsonable(v) for v in o]
        try:
            if _torch.is_tensor(o):
                o = o.detach().cpu().numpy()
        except Exception:
            pass
        if isinstance(o, _np.ndarray):
            return _make_jsonable(o.tolist())
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.integer,)):
            return int(o)
        return o

    with open(metrics_path, 'w') as f:
        json.dump(_make_jsonable(metrics), f, indent=2)

    del model, criterion, optimizer, scheduler, scaler, train_loader, val_loader, heldout_loader, preds, trues, nz_masks, meta
    cleanup_cuda(device)
    return metrics



def _run_gpu_worker(device_id, tasks, position, results_lock, all_metrics):
    worker_metrics = []
    for variant, seed in tasks:
        print(f"\n=== GPU {device_id} | {variant['label']} | seed {seed} ===")
        try:
            metrics = fit_one_run(variant, seed, device_id=device_id, position=position)
        except RuntimeError as e:
            cleanup_cuda(torch.device(f'cuda:{device_id}') if device_id is not None and torch.cuda.is_available() else None)
            if 'out of memory' in str(e).lower():
                print('CUDA OOM. Try lowering BATCH_SIZE to 64 or use fewer concurrent GPU workers, then rerun from the config cell.')
            raise
        worker_metrics.append(metrics)
        with results_lock:
            all_metrics.append(metrics)
            pd.DataFrame(all_metrics).to_csv(Path(TRAJ_CONFIG['save_dir']) / 'ablation_results_partial.csv', index=False)
    return worker_metrics



def run_ablation_grid(variants=ABLATION_VARIANTS, seeds=RUN_SEEDS):
    tasks = [(variant, seed) for variant in variants for seed in seeds]
    if torch.cuda.is_available() and CUDA_DEVICE_IDS:
        gpu_ids = list(CUDA_DEVICE_IDS)
    else:
        gpu_ids = [None]

    task_chunks = {gpu_id: [] for gpu_id in gpu_ids}
    for idx, task in enumerate(tasks):
        task_chunks[gpu_ids[idx % len(gpu_ids)]].append(task)

    all_metrics = []
    results_lock = Lock()
    max_workers = len(gpu_ids)
    print(f'Launching {max_workers} worker(s): one model per GPU, batch size {BATCH_SIZE} per model.')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_gpu_worker, gpu_id, task_chunks[gpu_id], pos, results_lock, all_metrics)
            for pos, gpu_id in enumerate(gpu_ids)
            if task_chunks[gpu_id]
        ]
        for future in as_completed(futures):
            future.result()

    results_df = pd.DataFrame(all_metrics)
    if not results_df.empty:
        results_df = results_df.sort_values(['variant', 'seed']).reset_index(drop=True)
    results_df.to_csv(Path(TRAJ_CONFIG['save_dir']) / 'ablation_results.csv', index=False)
    return results_df



import argparse


def run_worker_tasks(tasks, physical_gpu_id, worker_index):
    variant_by_name = {variant['name']: variant for variant in ABLATION_VARIANTS}
    worker_metrics = []
    worker_dir = Path(TRAJ_CONFIG['save_dir']) / 'worker_outputs'
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_csv = worker_dir / f'worker_gpu_{physical_gpu_id}.csv'
    visible_device_id = CUDA_DEVICE_IDS[0] if CUDA_DEVICE_IDS else None
    print(f'Worker {worker_index} physical GPU {physical_gpu_id}; visible CUDA ids: {CUDA_DEVICE_IDS}', flush=True)
    print(f'Assigned tasks: {tasks}', flush=True)
    for variant_name, seed in tasks:
        variant = variant_by_name[variant_name]
        print(f'\n=== physical GPU {physical_gpu_id} | {variant["label"]} | seed {seed} ===', flush=True)
        metrics = fit_one_run(variant, int(seed), device_id=visible_device_id, position=0)
        metrics['physical_gpu_id'] = physical_gpu_id
        metrics['worker_index'] = worker_index
        worker_metrics.append(metrics)
        pd.DataFrame(worker_metrics).to_csv(worker_csv, index=False)
    return pd.DataFrame(worker_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-json', required=True)
    parser.add_argument('--physical-gpu-id', required=True)
    parser.add_argument('--worker-index', type=int, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.tasks_json)
    run_worker_tasks(tasks, args.physical_gpu_id, args.worker_index)

