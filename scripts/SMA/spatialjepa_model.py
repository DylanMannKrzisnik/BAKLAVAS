"""Shared SpatialJEPA model machinery.

This module holds the GCN-encoder surgery and (re)construction helpers shared
between the training script (``3__spatial_meta_modelling.py``) and the target
projection script (``4__project_target_data_SMA.py``). Keeping them here gives a
single source of truth for how a ``ConditionalVAESTSM`` is turned into a
spatial (teacher) or graph-free (student) SpatialJEPA model.
"""

from copy import deepcopy
from typing import Iterable, Optional, Union

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
from torch.distributions import kl_divergence as kld

import spatialmeta as smt
from spatialmeta.util.compat import Literal
from spatialmeta.util.logger import get_tqdm
from spatialmeta.util.loss import LossFunction


def build_spatial_edge_index(
    coords: np.ndarray,
    n_neighbors: int = 6,
    device: str = "cpu",
    section_labels=None,
):
    """kNN graph on spot coordinates for PyG GCNConv (spots = nodes).

    With ``section_labels`` (one label per row of ``coords``), the kNN graph is
    built **per section** and unioned into a single block-diagonal ``edge_index``
    (global node indexing, no edges between sections). This is required for
    horizontal integration: Visium pixel coordinates overlap across slides, so a
    single kNN over the pooled coordinates would create spurious cross-section
    edges.
    """
    from sklearn.neighbors import kneighbors_graph
    from torch_geometric.utils import from_scipy_sparse_matrix

    def _section_edges(sub_coords):
        k = min(n_neighbors, sub_coords.shape[0] - 1)
        if k < 1:
            return torch.empty((2, 0), dtype=torch.long)
        adj = kneighbors_graph(
            sub_coords,
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
        )
        adj = adj.maximum(adj.T)
        edge_index, _ = from_scipy_sparse_matrix(adj.tocoo())
        return edge_index.long()

    coords = np.asarray(coords)
    if section_labels is None:
        return _section_edges(coords).to(device)

    section_labels = np.asarray(section_labels)
    global_idx = np.arange(coords.shape[0])
    edge_blocks = []
    for section in np.unique(section_labels):
        mask = section_labels == section
        local_edges = _section_edges(coords[mask])
        if local_edges.numel() == 0:
            continue
        # map per-section local node indices back to global row indices
        local_to_global = torch.as_tensor(global_idx[mask], dtype=torch.long)
        edge_blocks.append(local_to_global[local_edges])
    if not edge_blocks:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(edge_blocks, dim=1).to(device)


def build_identity_edge_index(
    num_nodes: int,
    device: str = "cpu",
):
    """Self-loop-only graph: each node only sends messages to itself."""
    node_idx = torch.arange(num_nodes, dtype=torch.long, device=device)
    return torch.stack((node_idx, node_idx), dim=0)


def replace_fc_encoder_with_gcn(sae, edge_index, num_nodes: int, layer_idx: int = 0) -> None:
    """Replace one SAE encoder FCLayer with GCNConv + the same BN/ReLU/Dropout tail."""
    from torch_geometric.nn import GCNConv

    enc_idx = layer_idx * 2
    fc_layer = sae.layers[enc_idx]
    if fc_layer is None:
        raise ValueError(f"encoder layer {layer_idx} is missing")

    post_modules = [
        module for module in fc_layer._fclayer if not isinstance(module, nn.Linear)
    ]

    class GCNEncoderLayer(nn.Module):
        """GCNConv drop-in replacement for SpatialMETA's encoder FCLayer."""

        def __init__(self, fc, edge_index_, num_nodes, post):
            super().__init__()
            self.in_dim = fc.in_dim
            self.out_dim = fc.out_dim
            self.device = fc.device
            self.num_nodes = num_nodes
            self.register_buffer("edge_index", edge_index_)
            self.gcn = GCNConv(self.in_dim, self.out_dim, bias=True)
            self.post = post
            self._batch_node_idx = None

        def set_batch_node_idx(self, node_idx):
            self._batch_node_idx = node_idx

        def _gcn_edge_index(self, x):
            if x.size(0) == self.num_nodes:
                return self.edge_index
            if self._batch_node_idx is None:
                raise RuntimeError(
                    "GCN mini-batch requires node indices; use n_per_batch=n_obs "
                    "or patch_model_encode_for_gcn()."
                )
            from torch_geometric.utils import subgraph

            node_idx = self._batch_node_idx.to(x.device)
            sub_edge_index, _ = subgraph(
                node_idx,
                self.edge_index,
                relabel_nodes=True,
                num_nodes=self.num_nodes,
            )
            return sub_edge_index

        def forward(self, x, cat_list=None):
            if cat_list is not None:
                raise NotImplementedError(
                    "GCN encoder replacement does not support categorical inputs."
                )
            return self.post(self.gcn(x, self._gcn_edge_index(x)))

        def to(self, device):
            super().to(device)
            self.device = device
            return self

    sae.layers[enc_idx] = GCNEncoderLayer(
        fc_layer,
        edge_index,
        num_nodes=num_nodes,
        post=nn.Sequential(*post_modules),
    )


def patch_model_encode_for_gcn(model) -> None:
    """Track mini-batch node indices so GCN encoder layers can build subgraphs."""

    class IndexTrackingLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            try:
                for batch in self.loader:
                    model._current_batch_indices = batch[0].detach().cpu()
                    yield batch
            finally:
                model._current_batch_indices = None

        def __len__(self):
            return len(self.loader)

    def _set_gcn_batch_indices(node_idx):
        if node_idx is not None:
            node_idx = torch.as_tensor(node_idx, dtype=torch.long)
        for encoder in (model.encoder_ST, model.encoder_SM):
            layer = encoder.layers[0]
            if hasattr(layer, "set_batch_node_idx"):
                layer.set_batch_node_idx(node_idx)

    _orig_as_dataloader = model.as_dataloader

    def as_dataloader_with_index_tracking(*args, **kwargs):
        return IndexTrackingLoader(_orig_as_dataloader(*args, **kwargs))

    model.as_dataloader = as_dataloader_with_index_tracking

    _orig_encode = model.encode

    def encode_with_gcn_indices(X, *args, **kwargs):
        node_idx = None
        if X.size(0) != model._n_record:
            node_idx = getattr(model, "_current_batch_indices", None)
            if node_idx is None:
                raise RuntimeError(
                    "GCN mini-batch encode requires original node indices. "
                    "Call through model.as_dataloader() or use n_per_batch=model._n_record."
                )
            if len(node_idx) != X.size(0):
                raise RuntimeError(
                    f"GCN batch index length ({len(node_idx)}) does not match "
                    f"input batch size ({X.size(0)})."
                )
        _set_gcn_batch_indices(node_idx)
        try:
            return _orig_encode(X, *args, **kwargs)
        finally:
            _set_gcn_batch_indices(None)

    model.encode = encode_with_gcn_indices


class ConditionalVAESTSM_STtoSM(smt.model.ConditionalVAESTSM):
    """``ConditionalVAESTSM`` with an added ST->SM cross-reconstruction objective.

    The base model only decodes each modality from its own latent
    (``decoder_st(z_st)``, ``decoder_sm(z_sm)``). For RNA-only transfer (e.g. SEA-AD)
    we additionally want ``decoder_sm(z_st)`` -- reconstructing SM from the ST-only
    expert latent ``H['st']['q_mu']`` -- to be *directly* supervised, since that is
    exactly the path used at inference (see
    ``embed_with_spatialmeta.get_st_latent_and_msi_decoding``). This reuses the
    existing ``decoder_sm`` / ``px_sm_*`` modules and adds **no new parameters**, so
    checkpoints remain interchangeable with the base class.
    """

    def _sm_cross_reconstruction_loss(self, X, H, batch_index, reduction):
        z_st = H["st"]["q_mu"]
        if batch_index is not None:
            z_st = torch.hstack([z_st, batch_index])
        px_sm_cross = self.decoder_sm(z_st.to(self.device))
        px_sm_cross_scale = self.px_sm_scale_decoder(px_sm_cross)
        px_sm_cross_rate = self.px_sm_rate_decoder(px_sm_cross)
        px_sm_cross_dropout = self.px_sm_dropout_decoder(px_sm_cross)
        X_SM = X[:, self._type == "SM"]
        if self.reconstruction_method_sm == "zg":
            return LossFunction.zi_gaussian_reconstruction_loss(
                X_SM,
                mean=px_sm_cross_scale,
                variance=px_sm_cross_rate.exp(),
                gate_logits=px_sm_cross_dropout,
                reduction=reduction,
            )
        elif self.reconstruction_method_sm == "mse":
            return nn.MSELoss()(px_sm_cross_scale, X_SM)
        else:  # 'g'
            return LossFunction.gaussian_reconstruction_loss(
                X_SM,
                mean=px_sm_cross_scale,
                variance=px_sm_cross_rate.exp(),
                reduction=reduction,
            )

    def forward(self, X, batch_index=None, reduction="sum"):
        H, Rs, L = super().forward(X, batch_index=batch_index, reduction=reduction)
        L["reconstruction_loss_sm_cross"] = self._sm_cross_reconstruction_loss(
            X, H, batch_index, reduction
        )
        return H, Rs, L

    def fit(
        self,
        max_epoch: int = 35,
        n_per_batch: int = 128,
        mode: Optional[Literal["single", "multi"]] = None,
        **kwargs,
    ):
        """Same presets as the base class plus a default ST->SM cross weight."""
        if mode == "single":
            kwargs["reconstruction_st_weight"] = 5
            kwargs["reconstruction_sm_weight"] = 1
            kwargs["reconstruction_st_corr_weight"] = 5
            kwargs["reconstruction_sm_corr_weight"] = 1
            kwargs["reconstruction_sm_cross_weight"] = 1
            kwargs["kl_weight"] = 0.5
        elif mode == "multi":
            kwargs["reconstruction_st_weight"] = 8
            kwargs["reconstruction_sm_weight"] = 2
            kwargs["reconstruction_st_corr_weight"] = 8
            kwargs["reconstruction_sm_corr_weight"] = 2
            kwargs["reconstruction_sm_cross_weight"] = 2
            kwargs["kl_weight"] = 1
            kwargs["mmd_weight"] = 10
        return self.fit_core(max_epoch=max_epoch, n_per_batch=n_per_batch, **kwargs)

    def fit_core(
        self,
        max_epoch: int = 35,
        n_per_batch: int = 128,
        reconstruction_reduction: str = "sum",
        kl_weight: float = 1.0,
        reconstruction_st_weight: float = 1.0,
        reconstruction_sm_weight: float = 1.0,
        reconstruction_st_corr_weight: float = 1.0,
        reconstruction_sm_corr_weight: float = 1.0,
        reconstruction_sm_cross_weight: float = 1.0,
        n_epochs_kl_warmup: Union[int, None] = 400,
        optimizer_parameters: Iterable = None,
        weight_decay: float = 1e-6,
        lr: bool = 5e-5,
        random_seed: int = 12,
        kl_loss_reduction: str = "mean",
        mmd_weight: float = 1.0,
    ):
        """Copy of ``ConditionalVAESTSM.fit_core`` with the added ST->SM cross term."""
        self.train()
        if n_epochs_kl_warmup:
            n_epochs_kl_warmup = min(max_epoch, n_epochs_kl_warmup)
            kl_warmup_gradient = kl_weight / n_epochs_kl_warmup
            kl_weight_max = kl_weight
            kl_weight = 0.0

        if optimizer_parameters is None:
            optimizer = optim.AdamW(self.parameters(), lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(optimizer_parameters, lr, weight_decay=weight_decay)
        pbar = get_tqdm()(range(max_epoch), desc="Epoch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        epoch_reconstruction_loss_st_list = []
        epoch_reconstruction_loss_sm_list = []
        epoch_reconstruction_loss_st_corr_list = []
        epoch_reconstruction_loss_sm_corr_list = []
        epoch_reconstruction_loss_sm_cross_list = []
        epoch_kldiv_loss_list = []
        epoch_total_loss_list = []
        epoch_mmd_loss_list = []

        for epoch in range(1, max_epoch + 1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss_sm = 0
            epoch_reconstruction_loss_st = 0
            epoch_reconstruction_loss_sm_corr = 0
            epoch_reconstruction_loss_st_corr = 0
            epoch_reconstruction_loss_sm_cross = 0
            epoch_kldiv_loss = 0
            epoch_mmd_loss = 0

            X_train = self.as_dataloader(batch_size=n_per_batch, shuffle=True)
            for batch_idx in X_train:
                indices = batch_idx[0].cpu().numpy()
                X_batch = []
                for idx in indices:
                    if scipy.sparse.issparse(self.X):
                        x_row = self.X.getrow(idx).toarray().squeeze()
                    else:
                        x_row = self.X[idx]
                    X_batch.append(x_row)
                X_batch = torch.tensor(np.stack(X_batch), dtype=torch.float32).to(self.device)
                if self.batch_codes is not None:
                    batch_index = [
                        torch.tensor(code[indices], dtype=torch.long).unsqueeze(1).to(self.device)
                        for code in self.batch_codes
                    ]
                    batch_index = torch.hstack(batch_index)
                else:
                    batch_index = None

                H, Rs, L = self.forward(
                    X_batch,
                    batch_index=batch_index,
                    reduction=reconstruction_reduction,
                )

                reconstruction_loss_st = L["reconstruction_loss_st"]
                reconstruction_loss_sm = L["reconstruction_loss_sm"]
                reconstruction_loss_st_corr = L["reconstruction_loss_st_corr"]
                reconstruction_loss_sm_corr = L["reconstruction_loss_sm_corr"]
                reconstruction_loss_sm_cross = L["reconstruction_loss_sm_cross"]
                kldiv_loss = L["kldiv_loss"]
                mmd_loss = L["mmd_loss"]

                avg_reconstruction_loss_st = reconstruction_loss_st.mean() / n_per_batch
                avg_reconstruction_loss_sm = reconstruction_loss_sm.mean() / n_per_batch
                avg_reconstruction_loss_st_corr = reconstruction_loss_st_corr.mean() / n_per_batch
                avg_reconstruction_loss_sm_corr = reconstruction_loss_sm_corr.mean() / n_per_batch
                avg_reconstruction_loss_sm_cross = reconstruction_loss_sm_cross.mean() / n_per_batch
                avg_mmd_loss = mmd_loss.mean() / n_per_batch

                if kl_loss_reduction == "mean":
                    avg_kldiv_loss = kldiv_loss.mean() / n_per_batch
                elif kl_loss_reduction == "sum":
                    avg_kldiv_loss = kldiv_loss.sum() / n_per_batch

                loss = (
                    (avg_reconstruction_loss_sm * reconstruction_sm_weight)
                    + (avg_reconstruction_loss_st * reconstruction_st_weight)
                    + (avg_reconstruction_loss_sm_corr * reconstruction_sm_corr_weight)
                    + (avg_reconstruction_loss_st_corr * reconstruction_st_corr_weight)
                    + (avg_reconstruction_loss_sm_cross * reconstruction_sm_cross_weight)
                    + (avg_kldiv_loss * kl_weight)
                    + (avg_mmd_loss * mmd_weight)
                )

                epoch_reconstruction_loss_sm += avg_reconstruction_loss_sm.item()
                epoch_reconstruction_loss_st += avg_reconstruction_loss_st.item()
                epoch_reconstruction_loss_sm_corr += avg_reconstruction_loss_sm_corr.item()
                epoch_reconstruction_loss_st_corr += avg_reconstruction_loss_st_corr.item()
                epoch_reconstruction_loss_sm_cross += avg_reconstruction_loss_sm_cross.item()
                epoch_mmd_loss += avg_mmd_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                epoch_total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.set_postfix(
                {
                    "reconst_sm": "{:.2e}".format(epoch_reconstruction_loss_sm),
                    "reconst_st": "{:.2e}".format(epoch_reconstruction_loss_st),
                    "reconst_sm_corr": "{:.2e}".format(epoch_reconstruction_loss_sm_corr),
                    "reconst_st_corr": "{:.2e}".format(epoch_reconstruction_loss_st_corr),
                    "reconst_sm_cross": "{:.2e}".format(epoch_reconstruction_loss_sm_cross),
                    "kldiv": "{:.2e}".format(epoch_kldiv_loss),
                    "total_loss": "{:.2e}".format(epoch_total_loss),
                    "mmd_loss": "{:.2e}".format(epoch_mmd_loss),
                }
            )

            pbar.update(1)
            epoch_reconstruction_loss_sm_list.append(epoch_reconstruction_loss_sm)
            epoch_reconstruction_loss_st_list.append(epoch_reconstruction_loss_st)
            epoch_reconstruction_loss_sm_corr_list.append(epoch_reconstruction_loss_sm_corr)
            epoch_reconstruction_loss_st_corr_list.append(epoch_reconstruction_loss_st_corr)
            epoch_reconstruction_loss_sm_cross_list.append(epoch_reconstruction_loss_sm_cross)
            epoch_kldiv_loss_list.append(epoch_kldiv_loss)
            epoch_total_loss_list.append(epoch_total_loss)
            epoch_mmd_loss_list.append(epoch_mmd_loss)

            if n_epochs_kl_warmup:
                kl_weight = min(kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1

        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())

        return dict(
            epoch_reconstruction_loss_st_list=epoch_reconstruction_loss_st_list,
            epoch_reconstruction_loss_sm_list=epoch_reconstruction_loss_sm_list,
            epoch_reconstruction_loss_st_corr_list=epoch_reconstruction_loss_st_corr_list,
            epoch_reconstruction_loss_sm_corr_list=epoch_reconstruction_loss_sm_corr_list,
            epoch_reconstruction_loss_sm_cross_list=epoch_reconstruction_loss_sm_cross_list,
            epoch_kldiv_loss_list=epoch_kldiv_loss_list,
            epoch_total_loss_list=epoch_total_loss_list,
            epoch_mmd_loss_list=epoch_mmd_loss_list,
        )


DECODER_MODULE_NAMES = (
    "decoder_st",
    "decoder_sm",
    "px_rna_scale_decoder",
    "px_rna_rate_decoder",
    "px_rna_dropout_decoder",
    "px_sm_scale_decoder",
    "px_sm_rate_decoder",
    "px_sm_dropout_decoder",
)


def _module_has_graph_op(module) -> bool:
    """True if any submodule is a graph (message-passing) layer."""
    from torch_geometric.nn import MessagePassing

    return any(
        isinstance(m, MessagePassing) or type(m).__name__ == "GCNEncoderLayer"
        for m in module.modules()
    )


def copy_decoder_weights(teacher, student):
    """Copy teacher decode-side weights into student, skipping any graph-op module.

    The student's decoder is never trained by ``SpatialJEPA_trainer`` (distillation
    only touches the encoder/latent heads), so without this its MSI decoding runs
    through random-init weights. Decoder modules are plain ``FCLayer``/``nn.Linear``
    today, so all of ``DECODER_MODULE_NAMES`` are copied; any module that grows a
    graph op is skipped (returned in ``skipped``) since its weights are not a safe
    drop-in for the graph-free student.
    """
    copied, skipped = [], []
    for name in DECODER_MODULE_NAMES:
        t = getattr(teacher, name, None)
        s = getattr(student, name, None)
        if t is None or s is None:
            continue
        if _module_has_graph_op(t) or _module_has_graph_op(s):
            skipped.append(name)
            continue
        s.load_state_dict(t.state_dict())
        copied.append(name)
    return copied, skipped


def spatialJEPA_model(
    joint_adata,
    graph_conv=True,
    full_graph=False,
    device="cuda:0",
    batch_keys=None,
    section_key=None,
):
    """Build a ConditionalVAESTSM, optionally with GCN encoders.

    ``joint_adata`` must carry spot coordinates in ``obsm['spatial']`` when
    ``graph_conv=True`` (full-graph teacher needs them to build the kNN graph;
    the identity-graph student only needs ``n_obs``).

    For horizontal (multi-section) integration, pass ``batch_keys`` (a list of
    ``obs`` columns identifying the section/batch) to enable decoder batch
    conditioning + the MMD alignment loss, and ``section_key`` so the full-graph
    teacher builds a block-diagonal spatial graph (no cross-section edges).
    """

    model = ConditionalVAESTSM_STtoSM(
        joint_adata,
        device=device,
        reconstruction_method_sm='g',
        reconstruction_method_st='zinb',
        batch_keys=batch_keys,
    )

    if graph_conv:
        if "spatial" not in joint_adata.obsm:
            raise ValueError(
                "graph_conv=True requires joint_adata.obsm['spatial']; set it before "
                "calling spatialJEPA_model()."
            )

        if full_graph:
            section_labels = (
                joint_adata.obs[section_key].to_numpy()
                if section_key is not None
                else None
            )
            edge_index = build_spatial_edge_index(
                joint_adata.obsm["spatial"],
                n_neighbors=6,
                device=str(model.device),
                section_labels=section_labels,
            )
        else:
            edge_index = build_identity_edge_index(
                joint_adata.n_obs,
                device=str(model.device),
            )

        # SAE.layers = ModuleList([FCLayer(in->128), None]) for encode_only stacks=[128]
        num_nodes = joint_adata.n_obs
        replace_fc_encoder_with_gcn(model.encoder_ST, edge_index, num_nodes, layer_idx=0)
        replace_fc_encoder_with_gcn(model.encoder_SM, edge_index, num_nodes, layer_idx=0)
        patch_model_encode_for_gcn(model)
        model.to(model.device)

    return model


def save_spatialjepa_model(model, path, *, full_graph: bool):
    """Persist a SpatialJEPA model plus the metadata needed to rebuild it."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "full_graph": full_graph,
            "graph_conv": True,
            "reconstruction_method_sm": model.reconstruction_method_sm,
            "reconstruction_method_st": model.reconstruction_method_st,
            "hidden_stacks": model.hidden_stacks,
            "n_latent": model.n_latent,
            "batch_keys": getattr(model, "batch_keys", None),
        },
        path,
    )


def load_spatialjepa_model(checkpoint_path, template_joint_adata, device="cuda:0"):
    """Rebuild a saved SpatialJEPA model on a (possibly new) ``template_joint_adata``.

    The checkpoint's ``edge_index`` buffers carry the *training* graph (sized to
    the training ``n_obs``); they are dropped so the freshly built, target-sized
    graph from ``spatialJEPA_model`` is kept instead. All learned weights load by
    name via ``strict=False``.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = spatialJEPA_model(
        template_joint_adata,
        graph_conv=ckpt["graph_conv"],
        full_graph=ckpt["full_graph"],
        device=device,
        batch_keys=ckpt.get("batch_keys"),
    )
    state = {
        k: v for k, v in ckpt["state_dict"].items() if not k.endswith("edge_index")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    leftover = [k for k in (list(missing) + list(unexpected)) if not k.endswith("edge_index")]
    if leftover:
        raise RuntimeError(
            f"Unexpected missing/unexpected keys when loading {checkpoint_path}: {leftover}"
        )
    model.eval()
    return model
