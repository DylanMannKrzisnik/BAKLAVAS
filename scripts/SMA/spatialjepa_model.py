"""Shared SpatialJEPA model machinery.

This module holds the GCN-encoder surgery and (re)construction helpers shared
between the training script (``3__spatial_meta_modelling.py``) and the target
projection script (``4__project_target_data_SMA.py``). Keeping them here gives a
single source of truth for how a ``ConditionalVAESTSM`` is turned into a
spatial (teacher) or graph-free (student) SpatialJEPA model.
"""

import numpy as np
import torch
import torch.nn as nn

import spatialmeta as smt


def build_spatial_edge_index(
    coords: np.ndarray,
    n_neighbors: int = 6,
    device: str = "cpu",
):
    """kNN graph on spot coordinates for PyG GCNConv (spots = nodes)."""
    from sklearn.neighbors import kneighbors_graph
    from torch_geometric.utils import from_scipy_sparse_matrix

    adj = kneighbors_graph(
        coords,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
    )
    adj = adj.maximum(adj.T)
    edge_index, _ = from_scipy_sparse_matrix(adj.tocoo())
    return edge_index.to(device)


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


def spatialJEPA_model(joint_adata, graph_conv=True, full_graph=False, device="cuda:0"):
    """Build a ConditionalVAESTSM, optionally with GCN encoders.

    ``joint_adata`` must carry spot coordinates in ``obsm['spatial']`` when
    ``graph_conv=True`` (full-graph teacher needs them to build the kNN graph;
    the identity-graph student only needs ``n_obs``).
    """

    model = smt.model.ConditionalVAESTSM(
        joint_adata,
        device=device,
        reconstruction_method_sm='g',
        reconstruction_method_st='zinb',
    )

    if graph_conv:
        if "spatial" not in joint_adata.obsm:
            raise ValueError(
                "graph_conv=True requires joint_adata.obsm['spatial']; set it before "
                "calling spatialJEPA_model()."
            )

        if full_graph:
            edge_index = build_spatial_edge_index(
                joint_adata.obsm["spatial"],
                n_neighbors=6,
                device=str(model.device),
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
