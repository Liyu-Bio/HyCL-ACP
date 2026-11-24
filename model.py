"""Model and feature definitions for HyCL-ACP."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent


SVM_C = 5.0


ESM_MODEL = "esm2_t33_650M_UR50D"


ESM_LAYER = 12


ESM_POOLING = "mean"


ESM_DIMENSIONS = 1280


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


AA_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


AAC_DIMENSIONS = len(AMINO_ACIDS)


DPC_DIMENSIONS = AAC_DIMENSIONS * AAC_DIMENSIONS


SEQUENCE_DESCRIPTOR_DIMENSIONS = AAC_DIMENSIONS + DPC_DIMENSIONS


ESM_BRANCH_DIMENSIONS = ESM_DIMENSIONS + SEQUENCE_DESCRIPTOR_DIMENSIONS


def resolve_data(path: Path) -> tuple[str, Path]:
    path = path.expanduser().resolve()
    if path.is_dir():
        train, test = path / "train.csv", path / "test.csv"
        if not train.exists() or not test.exists():
            raise ValueError(f"Binary directory must contain train.csv and test.csv: {path}")
        return "binary", path
    if path.is_file():
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        columns = set(pd.read_csv(path, sep=separator, nrows=1).columns)
        if {"text", "label"} <= columns:
            return "multilabel", path
    raise FileNotFoundError(path)


def dataset_key(data: Path) -> str:
    task, data = resolve_data(data)
    label = data.name if task == "binary" else data.stem
    label = __import__("re").sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower() or task
    digest = hashlib.sha256(str(data).encode()).hexdigest()[:10]
    return f"{label}_{digest}"


def binary_frames(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    task, data = resolve_data(data)
    if task != "binary":
        raise ValueError(f"Not a binary dataset directory: {data}")
    return pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")


def multilabel_frame(data: Path) -> pd.DataFrame:
    """Load a multi-label dataset from TSV, TXT, or CSV."""
    _, data = resolve_data(data)
    separator = "\t" if data.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(data, sep=separator, dtype=str, keep_default_na=False)


def aac_dpc_descriptor(sequence: str) -> np.ndarray:
    """Return normalized AAC20 and adjacent-DPC400 descriptors."""
    sequence = str(sequence).strip().upper()
    invalid = set(sequence) - set(AMINO_ACIDS)
    if not sequence or invalid:
        raise ValueError(
            "Expected a non-empty canonical amino-acid sequence; "
            f"invalid={sorted(invalid)}")

    length = len(sequence)
    aac = np.asarray([sequence.count(amino_acid) / length
                      for amino_acid in AMINO_ACIDS], dtype=np.float32)
    dpc = np.zeros(DPC_DIMENSIONS, dtype=np.float32)
    if length > 1:
        scale = 1.0 / (length - 1)
        for left, right in zip(sequence, sequence[1:]):
            dpc[AA_INDEX[left] * AAC_DIMENSIONS + AA_INDEX[right]] += scale
    return np.concatenate([aac, dpc])


def PSRA_aac_dpc_features(sequences: list[str]) -> np.ndarray:
    """Vectorize the 420-dimensional AAC+DPC block used by PSRA."""
    return np.stack([aac_dpc_descriptor(sequence) for sequence in sequences])


def cached_PSRA_esm_view(cache: np.lib.npyio.NpzFile, feature_name: str) -> np.ndarray:
    """Read the ESM-2 feature view used by PSRA."""
    if feature_name in cache:
        features = np.asarray(cache[feature_name], dtype=np.float32)
    elif "features" in cache:
        metadata = {
            "model_name": ESM_MODEL,
            "pooling": ESM_POOLING,
            "layer": ESM_LAYER,
        }
        for name, expected in metadata.items():
            if name not in cache or cache[name].item() != expected:
                raise RuntimeError(
                    f"Legacy ESM cache metadata mismatch for {name}: expected {expected!r}")
        features = np.asarray(cache["features"], dtype=np.float32)
    else:
        raise RuntimeError(f"ESM cache lacks required view: {feature_name}")
    if features.ndim != 2 or features.shape[1] != ESM_DIMENSIONS:
        raise RuntimeError(
            f"Expected a 2D {ESM_DIMENSIONS}-dimensional ESM view, got "
            f"shape {features.shape}")
    return features


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_PSRA_svm(c_value: float = SVM_C):
    """Build the unfitted class-balanced RBF-SVM used by PSRA."""
    return make_pipeline(StandardScaler(), SVC(
        C=c_value, kernel="rbf", probability=True,
        class_weight="balanced", random_state=5))


MODEL_SPECS = {
    "esm2_t6_8M_UR50D": ("esm2_t6_8M_UR50D", 6, [3]),
    "esm2_t12_35M_UR50D": ("esm2_t12_35M_UR50D", 12, [12]),
    "esm2_t30_150M_UR50D": ("esm2_t30_150M_UR50D", 30, [15]),
    "esm2_t33_650M_UR50D": ("esm2_t33_650M_UR50D", 33, [17]),
}


def normalized_sequences(values) -> list[str]:
    """Normalize and validate amino-acid sequences."""
    sequences = values.astype(str).str.strip().str.upper().tolist()
    invalid = [index for index, sequence in enumerate(sequences)
               if not sequence or set(sequence) - set(AMINO_ACIDS)]
    if invalid:
        raise ValueError(f"Invalid amino-acid sequences at rows {invalid[:10]}")
    return sequences


def sequence_digest(sequences: list[str]) -> str:
    """Return a stable digest for a sequence collection."""
    return hashlib.sha256("\n".join(sequences).encode()).hexdigest()


def esm_inputs(data: Path) -> list[tuple[str, list[str]]]:
    """Build named ESM extraction inputs for binary or multi-label data."""
    task, data = resolve_data(data)
    key = dataset_key(data)
    if task == "binary":
        train, test = binary_frames(data)
        return [(f"{key}_train", normalized_sequences(train["Feature"])),
                (f"{key}_test", normalized_sequences(test["Feature"]))]
    frame = multilabel_frame(data)
    return [(key, normalized_sequences(frame["text"]))]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, args):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(args.d_k)

        # Keep the masking operation compatible with AMP/FP16.
        scores = scores.masked_fill(
            attn_mask,
            torch.finfo(scores.dtype).min
        )

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.args.n_heads, self.args.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.args.n_heads, self.args.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.args.n_heads, self.args.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask, self.args)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.args.n_heads * self.args.d_v)
        output = self.fc(context)

        return F.layer_norm(output + residual, (self.args.d_model,)), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False),
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return F.layer_norm(output + residual, (self.args.d_model,))


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        # Token 0 represents padding.
        self.src_emb = nn.Embedding(
            args.src_vocab_size_seq,
            args.d_model,
            padding_idx=0
        )

        # Multi-scale local motif encoding.
        self.local_motif = MSCEConvolutionBlock(
            args.d_model,
            dropout=args.dropout
        )

        # Preserve the original positional encoding.
        self.pos_emb = PositionalEncoding(
            args.d_model
        )

        # Preserve the original Transformer configuration.
        self.layers = nn.ModuleList([
            EncoderLayer(args)
            for _ in range(args.n_layers)
        ])

    def forward(self, enc_inputs):

        # True denotes a valid amino-acid token.
        valid = enc_inputs != 0

        # =====================================
        # 1. Amino-acid embedding
        # =====================================
        enc_outputs = self.src_emb(enc_inputs)
        # [B, L, d_model]

        # =====================================
        # 2. Local motif enhancement
        # =====================================
        enc_outputs = self.local_motif(
            enc_outputs,
            valid
        )

        # =====================================
        # 3. Original positional encoding
        # =====================================
        enc_outputs = self.pos_emb(
            enc_outputs.transpose(0, 1)
        ).transpose(0, 1)

        # Positional encoding makes padded positions nonzero, so clear them again.
        enc_outputs = enc_outputs.masked_fill(
            ~valid.unsqueeze(-1),
            0.0
        )

        # =====================================
        # 4. Original Transformer
        # =====================================
        enc_self_attn_mask = get_attn_pad_mask(
            enc_inputs,
            enc_inputs
        )

        for layer in self.layers:
            enc_outputs, _ = layer(
                enc_outputs,
                enc_self_attn_mask
            )

            # Prevent padded tokens from accumulating information across layers.
            enc_outputs = enc_outputs.masked_fill(
                ~valid.unsqueeze(-1),
                0.0
            )

        return enc_outputs, None


class MSCEConvolutionBlock(nn.Module):
    """
    Multi-scale local peptide motif extractor.
    Input/output: [B, L, d_model].
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # Depthwise convolutions capture motifs spanning 3, 5, or 7 residues.
        self.conv3 = nn.Conv1d(
            d_model, d_model,
            kernel_size=3,
            padding=1,
            groups=d_model
        )

        self.conv5 = nn.Conv1d(
            d_model, d_model,
            kernel_size=5,
            padding=2,
            groups=d_model
        )

        self.conv7 = nn.Conv1d(
            d_model, d_model,
            kernel_size=7,
            padding=3,
            groups=d_model
        )

        # Fuse multi-scale motifs.
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Control how much local information is introduced.
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, valid):
        # Input x: [B, L, d].
        residual = x

        # [B, d, L]
        z = x.transpose(1, 2)

        x3 = self.conv3(z).transpose(1, 2)
        x5 = self.conv5(z).transpose(1, 2)
        x7 = self.conv7(z).transpose(1, 2)

        local = torch.cat([x3, x5, x7], dim=-1)
        local = self.fuse(local)

        gate = self.gate(
            torch.cat([residual, local], dim=-1)
        )

        out = residual + gate * local
        out = self.norm(out)

        # Explicitly clear padded positions.
        out = out.masked_fill(
            ~valid.unsqueeze(-1), 0.0
        )

        return out


SeqEncoder = Encoder


beads = {'Nda': 1, 'N0': 2, 'Qa': 4, 'Qd': 3, 'SC4': 5, 'SC5': 6, 'C3': 7, 'C5': 8,
         'AC1': 9, 'AC2': 10, 'P1': 11, 'P4': 12, 'P5': 13, 'SP1': 14, 'Snd': 0}


amino_acids_feature = {
    'W': [beads['Nda'], beads['SC4'], beads['Snd'], beads['SC5'], beads['SC5']],
    'F': [beads['Nda'], beads['SC5'], beads['SC5'], beads['SC5']],
    'Y': [beads['Nda'], beads['SC4'], beads['SC4'], beads['SP1']],
    'H': [beads['Nda'], beads['SC4'], beads['SP1'], beads['SP1']],
    'K': [beads['Nda'], beads['C3'], beads['Qd']],
    'R': [beads['Nda'], beads['N0'], beads['Qd']],
    'D': [beads['Nda'], beads['Qa']],
    'E': [beads['Nda'], beads['Qa']],
    'V': [beads['Nda'], beads['AC2']],
    'I': [beads['Nda'], beads['AC1']],
    'L': [beads['Nda'], beads['AC1']],
    'M': [beads['Nda'], beads['C5']],
    'C': [beads['Nda'], beads['C5']],
    'S': [beads['Nda'], beads['P1']],
    'T': [beads['Nda'], beads['P1']],
    'Q': [beads['Nda'], beads['P4']],
    'N': [beads['Nda'], beads['P5']],
    'P': [beads['N0'], beads['C3']],
    'A': [beads['N0']],
    'G': [beads['Nda']],
}


amino_acids_connection = {
    'W': [[0, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 1, 0, 1, 1], [0, 1, 1, 0, 1], [0, 0, 1, 1, 0]],
    'F': [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]],
    'Y': [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]],
    'H': [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]],
    'K': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    'R': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    'D': [[0, 1], [1, 0]],
    'E': [[0, 1], [1, 0]],
    'V': [[0, 1], [1, 0]],
    'I': [[0, 1], [1, 0]],
    'L': [[0, 1], [1, 0]],
    'M': [[0, 1], [1, 0]],
    'C': [[0, 1], [1, 0]],
    'S': [[0, 1], [1, 0]],
    'T': [[0, 1], [1, 0]],
    'Q': [[0, 1], [1, 0]],
    'N': [[0, 1], [1, 0]],
    'P': [[0, 1], [1, 0]],
    'A': [[0]],
    'G': [[0]],
}


def make_data(features):
    """Convert a list of peptide strings into DGL graphs."""
    dataset = []
    for i in range(len(features)):
        graph_attr = []
        residue_ids = []
        count = 0
        for n in list(str(features[i])):
            if count == 0:
                residue_beads = [beads['Qd']] + amino_acids_feature[n][1:]
            elif count == len(features[i]) - 1:
                residue_beads = [beads['Qa']] + amino_acids_feature[n][1:]
            else:
                residue_beads = amino_acids_feature[n]
            graph_attr.extend(residue_beads)
            residue_ids.extend([count] * len(residue_beads))
            count = count + 1

        graph_adja = np.zeros((len(graph_attr), len(graph_attr)))
        position = 0
        last_backbone = 0
        for n in list(str(features[i])):
            size = len(amino_acids_connection[n])
            graph_adja[position:position + size, position:position + size] = np.array(
                amino_acids_connection[n])
            # Connect adjacent backbone beads.
            if position:
                graph_adja[position, last_backbone] = 1
                graph_adja[last_backbone, position] = 1
            last_backbone = position
            position = position + size
        for n in range(len(graph_attr)):
            graph_adja[n, n] = 1
        g = nx.Graph(graph_adja)
        for node in range(g.number_of_nodes()):
            g.nodes[node]['feat'] = torch.tensor(graph_attr[node], dtype=torch.long)
            # Retain bead-to-residue assignments for graph-sequence alignment.
            g.nodes[node]['residue_id'] = torch.tensor(
                residue_ids[node], dtype=torch.long)
        g = dgl.from_networkx(g, node_attrs=['feat', 'residue_id'])
        dataset.append(g)
    return dataset


src_vocab = {'Empty': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
             'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
             'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}


def make_seq_data(features, src_len):
    """Convert peptide strings into a padded long tensor of shape [N, src_len]."""
    enc_inputs = []
    for i in range(len(features)):
        enc_input = [[src_vocab[n] for n in list(str(features[i]))]]
        while len(enc_input[0]) < src_len:
            enc_input[0].append(0)
        enc_inputs.extend(enc_input)
    return torch.LongTensor(enc_inputs)


def masked_attention_pool(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    scorer: nn.Module,
) -> torch.Tensor:
    """Apply learnable attention pooling over valid tokens.

    Args:
        tokens: Token representations with shape ``[B, N, d]``.
        valid: Valid-position mask with shape ``[B, N]``; True is valid.
        scorer: Learnable module that maps each token to a scalar score.

    Returns:
        Pooled batch representations with shape ``[B, d]``.
    """
    scores = scorer(tokens).squeeze(-1)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
    return (tokens * weights).sum(dim=1)


class MSCE(nn.Module):
    """Multi-scale Sequence Context Encoding from the HyCL-ACP framework."""

    def __init__(self, args):
        super().__init__()
        self.encoder = SeqEncoder(args)
        self.attn_pool = nn.Linear(args.d_model, 1)
        self.proj = nn.Sequential(
            nn.Linear(args.d_model, args.d_fuse),
            nn.LayerNorm(args.d_fuse),
        )

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence: Amino-acid indices with shape ``[B, L]``; 0 is padding.

        Returns:
            ``(sequence_tokens, sequence_vector, valid_mask)``.
        """
        tokens, _ = self.encoder(sequence)  # [B, L, d_model]
        valid = sequence != 0  # [B, L]

        sequence_vector = masked_attention_pool(
            tokens,
            valid,
            self.attn_pool,
        )
        sequence_vector = self.proj(sequence_vector)  # [B, d_fuse]
        sequence_tokens = self.proj(tokens)  # [B, L, d_fuse]
        return sequence_tokens, sequence_vector, valid


class CSTL(nn.Module):
    """Coarse-grained Structure Topology Learning from HyCL-ACP.

    One or more Martini beads for each amino acid are first aggregated within
    each residue through learnable attention, then a residue-position embedding
    is added. All residues are retained.
    """

    def __init__(self, args):
        super().__init__()
        self.src_emb = nn.Embedding(args.src_vocab_size_graph, args.d_graph)
        self.convlayers = nn.ModuleList(
            [
                SAGEConv(
                    args.d_graph,
                    args.d_graph,
                    args.GraphSAGE_aggregator,
                )
                for _ in range(args.conv_layers)
            ]
        )
        self.node_proj = nn.Sequential(
            nn.Linear(args.d_graph, args.d_fuse),
            nn.LayerNorm(args.d_fuse),
        )
        self.residue_pool = nn.Linear(args.d_fuse, 1)
        self.residue_position = nn.Embedding(args.src_len, args.d_fuse)
        self.residue_norm = nn.LayerNorm(args.d_fuse)
        self.attn_pool = nn.Linear(args.d_fuse, 1)

    def _pool_graph_residues(
        self,
        node_features: torch.Tensor,
        residue_ids: torch.Tensor,
        batch_node_counts: list[int],
    ) -> list[torch.Tensor]:
        """Aggregate beads from the same residue into one token per graph."""
        residue_sequences = []
        offset = 0

        for node_count in batch_node_counts:
            graph_features = node_features[offset : offset + node_count]
            graph_residue_ids = residue_ids[offset : offset + node_count]
            residue_count = int(graph_residue_ids.max().item()) + 1

            residue_tokens = []
            for residue_index in range(residue_count):
                members = graph_features[graph_residue_ids == residue_index]
                scores = self.residue_pool(members).squeeze(-1)
                weights = torch.softmax(scores, dim=0).unsqueeze(-1)
                token = (members * weights).sum(dim=0)

                position = torch.tensor(
                    residue_index,
                    dtype=torch.long,
                    device=node_features.device,
                )
                token = self.residue_norm(
                    token + self.residue_position(position)
                )
                residue_tokens.append(token)

            residue_sequences.append(torch.stack(residue_tokens, dim=0))
            offset += node_count

        return residue_sequences

    @staticmethod
    def _pad_residue_sequences(
        residue_sequences: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad residue-token sequences and return a mask where True is padding."""
        max_residues = max(tokens.size(0) for tokens in residue_sequences)
        padded_sequences = []
        padding_masks = []

        for tokens in residue_sequences:
            residue_count = tokens.size(0)
            padding_count = max_residues - residue_count

            if padding_count:
                padding = torch.zeros(
                    padding_count,
                    tokens.size(-1),
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                tokens = torch.cat([tokens, padding], dim=0)

            padding_mask = torch.zeros(
                max_residues,
                dtype=torch.bool,
                device=tokens.device,
            )
            padding_mask[residue_count:] = True
            padded_sequences.append(tokens)
            padding_masks.append(padding_mask)

        return (
            torch.stack(padded_sequences, dim=0),
            torch.stack(padding_masks, dim=0),
        )

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ``(graph_tokens, graph_padding_mask, graph_vector)``。
        """
        bead_ids = graph.ndata["feat"].reshape(-1).long()
        residue_ids = graph.ndata["residue_id"].reshape(-1).long()

        node_features = self.src_emb(bead_ids)
        for layer in self.convlayers:
            node_features = F.relu(layer(graph, node_features))
        node_features = self.node_proj(node_features)  # [N, d_fuse]

        batch_node_counts = graph.batch_num_nodes()
        if torch.is_tensor(batch_node_counts):
            batch_node_counts = batch_node_counts.tolist()

        residue_sequences = self._pool_graph_residues(
            node_features,
            residue_ids,
            batch_node_counts,
        )
        graph_tokens, graph_padding_mask = self._pad_residue_sequences(
            residue_sequences
        )
        graph_vector = masked_attention_pool(
            graph_tokens,
            ~graph_padding_mask,
            self.attn_pool,
        )
        return graph_tokens, graph_padding_mask, graph_vector


class CIFCrossAttention(nn.Module):
    """Bidirectional cross-attention used by Cross-modality Information Fusion.

    Each layer contains graph-to-sequence and sequence-to-graph attention,
    gated residual updates, and two independent feed-forward networks.
    """

    def __init__(self, args):
        super().__init__()
        dimension = args.d_fuse
        self.layers = nn.ModuleList()

        for _ in range(args.cross_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "g2s": nn.MultiheadAttention(
                            dimension,
                            args.fusion_heads,
                            dropout=args.dropout,
                            batch_first=False,
                        ),
                        "s2g": nn.MultiheadAttention(
                            dimension,
                            args.fusion_heads,
                            dropout=args.dropout,
                            batch_first=False,
                        ),
                        "ng": nn.LayerNorm(dimension),
                        "ns": nn.LayerNorm(dimension),
                        "gate_g": nn.Linear(dimension * 2, dimension),
                        "gate_s": nn.Linear(dimension * 2, dimension),
                        "ffg": nn.Sequential(
                            nn.Linear(dimension, dimension * 2),
                            nn.GELU(),
                            nn.Dropout(args.dropout),
                            nn.Linear(dimension * 2, dimension),
                        ),
                        "ffs": nn.Sequential(
                            nn.Linear(dimension, dimension * 2),
                            nn.GELU(),
                            nn.Dropout(args.dropout),
                            nn.Linear(dimension * 2, dimension),
                        ),
                        "ng2": nn.LayerNorm(dimension),
                        "ns2": nn.LayerNorm(dimension),
                        "drop": nn.Dropout(args.dropout),
                    }
                )
            )

    def forward(
        self,
        graph_tokens: torch.Tensor,
        graph_padding_mask: torch.Tensor,
        sequence_tokens: torch.Tensor,
        sequence_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ``(graph_vector, sequence_vector)`` after cross-attention.
        """
        graph_states = graph_tokens.transpose(0, 1)  # [N, B, d]
        sequence_states = sequence_tokens.transpose(0, 1)  # [L, B, d]
        sequence_padding_mask = ~sequence_valid_mask

        for block in self.layers:
            graph_update, _ = block["g2s"](
                graph_states,
                sequence_states,
                sequence_states,
                key_padding_mask=sequence_padding_mask,
            )
            sequence_update, _ = block["s2g"](
                sequence_states,
                graph_states,
                graph_states,
                key_padding_mask=graph_padding_mask,
            )

            updated_graph = block["ng"](
                graph_states + block["drop"](graph_update)
            )
            updated_sequence = block["ns"](
                sequence_states + block["drop"](sequence_update)
            )

            graph_gate = torch.sigmoid(
                block["gate_g"](
                    torch.cat([updated_graph, graph_states], dim=-1)
                )
            )
            sequence_gate = torch.sigmoid(
                block["gate_s"](
                    torch.cat([updated_sequence, sequence_states], dim=-1)
                )
            )
            graph_states = (
                graph_gate * updated_graph
                + (1 - graph_gate) * graph_states
            )
            sequence_states = (
                sequence_gate * updated_sequence
                + (1 - sequence_gate) * sequence_states
            )

            graph_states = block["ng2"](
                graph_states + block["drop"](block["ffg"](graph_states))
            )
            sequence_states = block["ns2"](
                sequence_states
                + block["drop"](block["ffs"](sequence_states))
            )

        graph_states = graph_states.transpose(0, 1)  # [B, N, d]
        sequence_states = sequence_states.transpose(0, 1)  # [B, L, d]

        valid_graph = (~graph_padding_mask).float().unsqueeze(-1)
        graph_vector = (graph_states * valid_graph).sum(dim=1)
        graph_vector = graph_vector / valid_graph.sum(dim=1).clamp(min=1e-6)

        minimum = torch.finfo(sequence_states.dtype).min
        masked_sequence = sequence_states.masked_fill(
            ~sequence_valid_mask.unsqueeze(-1),
            minimum,
        )
        sequence_vector = masked_sequence.max(dim=1).values
        return graph_vector, sequence_vector


class CIFFeatureFusion(nn.Module):
    """Fuse the four feature sources defined by CIF with attention."""

    def __init__(self, args):
        super().__init__()
        dimension = args.d_fuse
        self.proj = nn.Linear(dimension, dimension)
        self.score = nn.Linear(dimension, 1)
        self.norm = nn.LayerNorm(dimension)
        self.drop = nn.Dropout(args.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.proj(tokens))
        weights = torch.softmax(self.score(hidden), dim=1)  # [B, 4, 1]
        fused = (tokens * weights).sum(dim=1)
        fused = self.drop(self.norm(fused))
        return fused, weights.squeeze(-1)


class FusionPredictor(nn.Module):
    """Shared MLP that maps the fused vector to classification logits."""

    def __init__(self, args):
        super().__init__()
        dimension = args.d_fuse
        self.fc_in = nn.Linear(dimension, dimension)
        self.layers = nn.ModuleList(
            [
                nn.Linear(dimension, dimension)
                for _ in range(args.fc_layers)
            ]
        )
        self.predictor = nn.Linear(dimension, args.output_layer)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.leaky_relu(self.fc_in(features))
        for layer in self.layers:
            hidden = layer(F.leaky_relu(hidden))
        return self.predictor(F.leaky_relu(hidden))


class CrossModalNeuralPathway(nn.Module):
    """MSCE-CSTL-CIF neural pathway used within the HyCL-ACP framework.

    This class does not represent the complete HyCL-ACP framework. The full
    framework additionally contains the PSRA pathway and decision-level fusion.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seq_enc = MSCE(args)
        self.graph_enc = CSTL(args)
        self.cross = CIFCrossAttention(args)
        self.fusion = CIFFeatureFusion(args)
        self.predictor = FusionPredictor(args)

    # Paper-aligned accessors retain the original registered module names so
    # existing checkpoint parameter keys remain compatible.
    @property
    def MSCE_module(self) -> MSCE:
        return self.seq_enc

    @property
    def CSTL_module(self) -> CSTL:
        return self.graph_enc

    @property
    def CIF_cross_attention(self) -> CIFCrossAttention:
        return self.cross

    @property
    def CIF_feature_fusion(self) -> CIFFeatureFusion:
        return self.fusion

    @property
    def fusion_predictor(self) -> FusionPredictor:
        return self.predictor

    def forward(self, sequence: torch.Tensor, graph) -> dict[str, torch.Tensor]:
        """Encode both modalities, apply cross-attention, and return predictions."""
        sequence_tokens, sequence_before, sequence_valid = self.MSCE_module(
            sequence
        )
        graph_tokens, graph_padding, graph_before = self.CSTL_module(graph)
        graph_after, sequence_after = self.CIF_cross_attention(
            graph_tokens,
            graph_padding,
            sequence_tokens,
            sequence_valid,
        )

        # Sources: graph before/after CIF, then sequence before/after CIF.
        fusion_sources = torch.stack(
            [
                graph_before,
                graph_after,
                sequence_before,
                sequence_after,
            ],
            dim=1,
        )
        fused, fusion_weights = self.CIF_feature_fusion(fusion_sources)

        fused_logits = self.fusion_predictor(fused)
        sequence_logits = self.fusion_predictor(sequence_before)
        graph_logits = self.fusion_predictor(graph_before)

        if self.args.task_type == "Classification":
            output = F.log_softmax(fused_logits, dim=1)
            sequence_output = F.log_softmax(sequence_logits, dim=1)
            graph_output = F.log_softmax(graph_logits, dim=1)
        else:
            # BCEWithLogitsLoss consumes logits directly; apply sigmoid at inference.
            output = fused_logits
            sequence_output = sequence_logits
            graph_output = graph_logits

        return {
            "out": output,
            "out_seq": sequence_output,
            "out_graph": graph_output,
            "w": fusion_weights,
        }

