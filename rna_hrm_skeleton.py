"""
rna_hrm_skeleton.py
Runnable PyTorch skeleton for an RNA Hierarchical Reasoning Model (RNA_HRM).

Requirements:
    - Python 3.8+
    - PyTorch (tested with 1.10+)

Run:
    python rna_hrm_skeleton.py
"""

import math
import random
import os
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Utilities / Simple helpers
# ---------------------------
NUC_VOCAB = {"A": 0, "U": 1, "G": 2, "C": 3}
VOCAB_SIZE = len(NUC_VOCAB)


def rand_seq(length: int):
    return [random.choice(list(NUC_VOCAB.values())) for _ in range(length)]


def seq_to_onehot(seq_tensor: torch.LongTensor, vocab_size=VOCAB_SIZE):
    # seq_tensor: (L,)
    oh = torch.nn.functional.one_hot(seq_tensor, num_classes=vocab_size).float()
    return oh  # (L, vocab_size)


# ---------------------------
# Synthetic Dataset
# ---------------------------
class SyntheticRNADataset(Dataset):
    """
    Produces small synthetic RNA examples.

    Real dataset should supply:
      - sequence: LongTensor (L,)
      - pair_labels: FloatTensor (L, L) symmetric, 1 for paired bases, 0 otherwise
      - coords: FloatTensor (L, natoms, 3) real atomic coords (or reduced to C4'/P/C1' etc)
      - coarse_targets: FloatTensor motif frames or centroids (here simplified)
    """

    def __init__(self, n_examples=500, seq_len=60, natoms=8):
        super().__init__()
        self.n_examples = n_examples
        self.seq_len = seq_len
        self.natoms = natoms

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        L = self.seq_len
        seq = torch.LongTensor(rand_seq(L))
        # synthetic pair labels: make stems every 10 residues
        pair_labels = torch.zeros((L, L), dtype=torch.float32)
        for i in range(0, L // 10):
            a = i * 10 + 1
            b = L - (i * 10) - 2
            if a < b:
                pair_labels[a, b] = 1.0
                pair_labels[b, a] = 1.0
        # synthetic coords: (L, natoms, 3)
        coords = torch.randn(L, self.natoms, 3) * 2.0
        # coarse frames: for simplicity a centroid per 10-residue motif (M, 3)
        n_motifs = max(1, L // 10)
        coarse_targets = torch.randn(n_motifs, 3)
        sample = {
            "seq": seq,
            "pair_labels": pair_labels,
            "coords": coords,
            "coarse_targets": coarse_targets,
        }
        return sample


# ---------------------------
# Model components (lightweight)
# ---------------------------
class RNASeqEncoder(nn.Module):
    """Simple Transformer-based sequence encoder. Replace with pretrained LM as needed."""

    def __init__(self, d_model=128, nhead=8, n_layers=3, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, seq: torch.LongTensor) -> torch.Tensor:
        # seq: (L,)
        x = self.embed(seq) * math.sqrt(self.embed.embedding_dim)  # (L, d)
        # transformer expects seq_len, batch, dim; we'll treat batch dim later outside
        x = x.unsqueeze(1)  # (L, 1, d)
        x = self.transformer(x)  # (L,1,d)
        x = x.squeeze(1)  # (L,d)
        return self.output_proj(x)  # (L,d)


class SecondaryPredictor(nn.Module):
    """
    Predicts probabilistic pairing matrix from sequence embeddings.
    Simple implementation: pair-score = dot(res_i, res_j) processed through a small MLP.
    """

    def __init__(self, d_model=128, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq_feats: torch.Tensor) -> torch.Tensor:
        # seq_feats: (L, d)
        L, d = seq_feats.shape
        # create pairwise concatenation efficiently
        a = seq_feats.unsqueeze(1).expand(L, L, d)  # (L,L,d)
        b = seq_feats.unsqueeze(0).expand(L, L, d)  # (L,L,d)
        ab = torch.cat([a, b], dim=-1)  # (L,L,2d)
        scores = self.mlp(ab).squeeze(-1)  # (L,L)
        # symmetricize and sigmoids
        scores = (scores + scores.transpose(0, 1)) / 2.0
        pair_probs = torch.sigmoid(scores)
        return pair_probs


class MotifGNNPlaceholder(nn.Module):
    """
    Placeholder that ingests pair_probs and seq_feats and outputs motif features.
    Replace with real graph network that clusters stems/junctions and passes messages.
    """

    def __init__(self, seq_dim=128, motif_dim=64):
        super().__init__()
        self.pool = nn.Linear(seq_dim, motif_dim)

    def forward(self, seq_feats: torch.Tensor, pair_probs: torch.Tensor) -> torch.Tensor:
        # Very simple pooling: split sequence into chunks (motifs) and average features
        L, d = seq_feats.shape
        motif_size = max(1, L // 10)
        motifs = []
        for i in range(0, L, motif_size):
            chunk = seq_feats[i : i + motif_size].mean(dim=0)
            motifs.append(chunk)
        motifs = torch.stack(motifs, dim=0)  # (n_motifs, d)
        return self.pool(motifs)  # (n_motifs, motif_dim)


class CoarseLayoutNet(nn.Module):
    """
    Predict motif centroids (M, 3). Replace with frame/quaternion outputs if desired.
    """

    def __init__(self, motif_dim=64):
        super().__init__()
        self.head = nn.Linear(motif_dim, 3)

    def forward(self, motif_feats: torch.Tensor) -> torch.Tensor:
        # motif_feats: (M, motif_dim)
        return self.head(motif_feats)  # (M,3)


class EquivariantRefinerPlaceholder(nn.Module):
    """
    Placeholder refiner: maps fused features + coarse to per-residue per-atom coords.
    Replace with SE(3)-equivariant network / diffusion for real performance.
    """

    def __init__(self, seq_dim=128, natoms=8):
        super().__init__()
        self.natoms = natoms
        self.mlp = nn.Sequential(
            nn.Linear(seq_dim + 3, 256),
            nn.ReLU(),
            nn.Linear(256, natoms * 3),
        )

    def forward(self, seq_feats: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        L, d = seq_feats.shape
        # simple scheme: broadcast motif centroid to residues by repeating (toy)
        # In real model you'd map each residue to its motif and pass motif coords
        # Here we tile the first motif centroid
        motif_centroid = coarse[0].unsqueeze(0).expand(L, 3)  # (L,3)
        x = torch.cat([seq_feats, motif_centroid], dim=-1)  # (L, d+3)
        out = self.mlp(x)  # (L, natoms*3)
        coords = out.view(L, self.natoms, 3)
        return coords


# ---------------------------
# Whole-model wrapper
# ---------------------------
class RNA_HRM(nn.Module):
    def __init__(self, d_model=128, natoms=8):
        super().__init__()
        self.seq_encoder = RNASeqEncoder(d_model=d_model)
        self.secondary_net = SecondaryPredictor(d_model)
        self.motif_gnn = MotifGNNPlaceholder(seq_dim=d_model)
        self.coarse_layout = CoarseLayoutNet(motif_dim=64)
        self.refiner = EquivariantRefinerPlaceholder(seq_dim=d_model, natoms=natoms)

    def forward(self, seq: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # seq: (L,) ints
        seq_feats = self.seq_encoder(seq)  # (L, d)
        pair_probs = self.secondary_net(seq_feats)  # (L,L)
        motif_feats = self.motif_gnn(seq_feats, pair_probs)  # (M, motif_dim)
        coarse = self.coarse_layout(motif_feats)  # (M, 3)
        coords = self.refiner(seq_feats, coarse)  # (L, natoms, 3)
        return {
            "seq_feats": seq_feats,
            "pair_probs": pair_probs,
            "motifs": motif_feats,
            "coarse": coarse,
            "coords": coords,
        }


# ---------------------------
# Losses
# ---------------------------
def pairing_loss(pred_pair: torch.Tensor, true_pair: torch.Tensor) -> torch.Tensor:
    # binary cross entropy, mask diagonal
    L = pred_pair.shape[0]
    mask = 1.0 - torch.eye(L, device=pred_pair.device)
    bce = torch.nn.BCELoss(reduction="sum")
    return bce(pred_pair * mask, true_pair * mask) / (mask.sum() + 1e-8)


def atom_mse_loss(pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
    # pred_coords: (L, natoms, 3), true_coords same
    return torch.nn.functional.mse_loss(pred_coords, true_coords)


def frame_loss(pred_coarse: torch.Tensor, true_coarse: torch.Tensor) -> torch.Tensor:
    # simple centroid MSE (real version should use frame/quaternion alignment)
    # pred_coarse: (M_pred, 3), true_coarse: (M_true, 3)
    # If different sizes, match min(M_pred, M_true)
    m = min(pred_coarse.shape[0], true_coarse.shape[0])
    return torch.nn.functional.mse_loss(pred_coarse[:m], true_coarse[:m])


# ---------------------------
# Training routine
# ---------------------------
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        seq = batch["seq"].to(device)  # (L,)
        pair_labels = batch["pair_labels"].to(device)  # (L,L)
        coords = batch["coords"].to(device)  # (L, natoms, 3)
        coarse_targets = batch["coarse_targets"].to(device)  # (M, 3)

        optimizer.zero_grad()
        outputs = model(seq)
        loss_pair = pairing_loss(outputs["pair_probs"], pair_labels)
        loss_atom = atom_mse_loss(outputs["coords"], coords)
        loss_frame = frame_loss(outputs["coarse"], coarse_targets)
        # combine with weights (tunable)
        loss = 1.0 * loss_pair + 1.0 * loss_atom + 0.5 * loss_frame
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_epoch(model: nn.Module, dataloader: DataLoader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            pair_labels = batch["pair_labels"].to(device)
            coords = batch["coords"].to(device)
            coarse_targets = batch["coarse_targets"].to(device)

            outputs = model(seq)
            loss_pair = pairing_loss(outputs["pair_probs"], pair_labels)
            loss_atom = atom_mse_loss(outputs["coords"], coords)
            loss_frame = frame_loss(outputs["coarse"], coarse_targets)
            loss = 1.0 * loss_pair + 1.0 * loss_atom + 0.5 * loss_frame
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ---------------------------
# Main
# ---------------------------
def collate_fn(batch):
    # All synthetic examples have same length in this skeleton; if using variable lengths,
    # pad sequences and masks here. We return the dict of tensors.
    return {
        "seq": batch[0]["seq"],
        "pair_labels": batch[0]["pair_labels"],
        "coords": batch[0]["coords"],
        "coarse_targets": batch[0]["coarse_targets"],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # hyperparams
    seq_len = 60
    natoms = 8
    batch_size = 1  # we use batch=1 because seq encoder uses transformer without batch dim in skeleton
    epochs = 5
    lr = 1e-4

    # dataset
    train_ds = SyntheticRNADataset(n_examples=200, seq_len=seq_len, natoms=natoms)
    val_ds = SyntheticRNADataset(n_examples=40, seq_len=seq_len, natoms=natoms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = RNA_HRM(d_model=128, natoms=natoms).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        # checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))
            print("Saved best model.")

    print("Training finished. Best val loss:", best_val)


if __name__ == "__main__":
    main()
