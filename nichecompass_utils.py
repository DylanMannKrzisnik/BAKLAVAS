"""
Utilities for customizing NicheCompass components.

This module is intended to host project-specific subclasses or helpers that
override behavior from the NicheCompass package (e.g., custom VGPGAE forward).
"""

from __future__ import annotations

from typing import Literal

from torch_geometric.data import Data

from nichecompass.data import SpatialAnnTorchDataset, dataprocessors
from nichecompass.modules import VGPGAE

import torch
import scipy as sp
from anndata import AnnData
from torch_geometric.utils import add_self_loops, remove_self_loops
from nichecompass.data.utils import encode_labels, sparse_mx_to_sparse_tensor
from typing import List, Optional

# Isolate functions from dataprocessors to avoid circular imports
edge_level_split = dataprocessors.edge_level_split
node_level_split_mask = dataprocessors.node_level_split_mask
prepare_data = dataprocessors.prepare_data


class CustomVGPGAE(VGPGAE):
    """
    Project-specific VGPGAE with custom behavior.

    Override methods as needed; this is a safe default that calls the parent
    implementation while giving you a single place to modify logic.
    """

    def forward(self,
                data_batch: Data,
                decoder: Literal["graph", "omics"],
                use_only_active_gps: bool=False,
                return_agg_weights: bool=False,
                update_atac_dynamic_decoder_mask: bool=False) -> dict:
        """
        Forward pass of the VGPGAE module.

        Parameters
        ----------
        data_batch:
            PyG Data object containing either an edge-level batch if 
            ´decoder == graph´ or a node-level batch if ´decoder == omics´.
        decoder:
            Decoder to use for the forward pass. Either ´graph´ for edge
            reconstruction or ´omics´ for gene expression and (if specified)
            chromatin accessibility reconstruction.
        use_only_active_gps:
            If ´True´, use only active gene programs as input to decoder.
        return_agg_weights:
            If ´True´, also return the aggregation weights of the node label
            aggregator.
        update_atac_dynamic_decoder_mask:
            If ´True´, turn off the mapped peaks for genes that have been
            turned off in a gene program (set peak gp weights to 0).

        Returns
        ----------
        output:
            Dictionary containing reconstructed edge logits if
            ´decoder == graph´ or the parameters of the omics feature
            distributions if ´decoder == omics´, as well as ´mu´ and ´logstd´ 
            from the latent space distribution.
        """
        x = data_batch.x # dim: n_obs x n_omics_features
        edge_index = data_batch.edge_index # dim: 2 x n_edges (incl. all edges
                                           # of sampled graph)
        
        # Get index of sampled nodes for current batch (neighbors of sampled
        # nodes are also part of the batch for message passing layers but
        # should be excluded in backpropagation)
        if decoder == "omics":
            # ´data_batch´ will be a node batch and first node_batch_size
            # elements are the sampled nodes, leading to a dim of ´batch_idx´ of
            # ´node_batch_size´
            batch_idx = slice(None, data_batch.batch_size)
        elif decoder == "graph":
            # ´data_batch´ will be an edge batch with sampled positive and
            # negative edges of size ´edge_batch_size´ respectively. Each edge
            # has a source and target node, leading to a dim of ´batch_idx´ of
            # 4 * ´edge_batch_size´
            batch_idx = torch.cat((data_batch.edge_label_index[0],
                                   data_batch.edge_label_index[1]), 0)

        # Logarithmitize omics feature vector (only) for encoder input for
        # numerical stability. This will not affect node labels.
        if self.log_variational_:
            x_enc = torch.log(1 + x)
        else:
            x_enc = x
            
        # Get categorical covariates embedding
        if len(self.cat_covariates_cats_) > 0:
            cat_covariates_embeds = []
            for i in range(len(self.cat_covariates_embedders)):
                cat_covariates_embeds.append(self.cat_covariates_embedders[i](
                     data_batch.cat_covariates_cats[:, i]))
                self.cat_covariates_embed = torch.cat(
                    cat_covariates_embeds,
                    dim=1)
        else:
            self.cat_covariates_embed = None         

        output = {}
        
        # Use encoder to get latent distribution parameters for current batch
        # and reparameterization trick to get latent features (gp scores).
        # Filter for nodes in current batch
        encoder_outputs = self.encoder(
            x=x_enc,
            edge_index=edge_index,
            cat_covariates_embed=(self.cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        self.mu = encoder_outputs[0][batch_idx, :]
        self.logstd = encoder_outputs[1][batch_idx, :]
        output["mu"] = self.mu
        output["logstd"] = self.logstd
        z = self.reparameterize(self.mu, self.logstd)

        if use_only_active_gps:
            active_gp_mask = self.get_active_gp_mask()
            
            # Set gp scores of inactive gene programs to 0 to not affect 
            # graph decoder
            z[:, ~active_gp_mask] = 0                

        if decoder == "omics":
            with torch.no_grad():
                if self.training:
                    # Update running mean absolute gp scores using exponential
                    # moving average with momentum of 0.1
                    mean_abs_mu = self.mu.norm(p=1, dim=0) / self.mu.size(0)
                    self.running_mean_abs_mu = (
                        0.1 * mean_abs_mu + 0.9 * self.running_mean_abs_mu)
                    
                if use_only_active_gps:
                    # Set running mean abs mu of inactive gene programs to 0 for
                    # active gp determination
                    self.running_mean_abs_mu[~active_gp_mask] = 0  

                    # Set dynamic mask to 0 for all inactive gene programs to
                    # not affect omics decoders
                    self.target_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0
                    self.source_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0

                    if "atac" in self.modalities_:
                        self.target_atac_dynamic_decoder_mask[~active_gp_mask, :] = 0
                        self.source_atac_dynamic_decoder_mask[~active_gp_mask, :] = 0
                    
            # Determine which features should be reconstructed based on
            # static and dynamic masks (if a feature is not connected to any
            # node it should not be reconstructed to not influence softmax
            # activation outputs). This can happen when no add-on gene programs
            # are present or when gene programs are turned off.
            if self.n_addon_gp_ > 0:
                target_rna_decoder_static_mask = torch.cat(
                    (self.target_rna_decoder_mask,
                     self.target_rna_decoder_addon_mask), dim=0)
                source_rna_decoder_static_mask = torch.cat(
                    (self.source_rna_decoder_mask,
                     self.source_rna_decoder_addon_mask), dim=0)
            else:
                target_rna_decoder_static_mask = self.target_rna_decoder_mask
                source_rna_decoder_static_mask = self.source_rna_decoder_mask

            self.target_n_gps_per_gene = (
                target_rna_decoder_static_mask
                * self.target_rna_dynamic_decoder_mask
                ).sum(0)
            self.features_idx_dict_["target_reconstructed_rna_idx"] = (
                torch.nonzero(self.target_n_gps_per_gene)).flatten().tolist()

            self.source_n_gps_per_gene = (
                source_rna_decoder_static_mask
                * self.source_rna_dynamic_decoder_mask
                ).sum(0)
            self.features_idx_dict_["source_reconstructed_rna_idx"] = (
                torch.nonzero(self.source_n_gps_per_gene)).flatten().tolist()

            self.target_rna_theta_reconstructed = self.target_rna_theta[
                self.features_idx_dict_["target_reconstructed_rna_idx"]]
            self.source_rna_theta_reconstructed = self.source_rna_theta[
                self.features_idx_dict_["source_reconstructed_rna_idx"]]
                    
            output["node_labels"] = {}

            # Get rna and atac part from omics feature vector
            x_atac = x[:, self.n_output_genes_:]
            x = x[:, :self.n_output_genes_]
        
            # Compute aggregated neighborhood rna feature vector
            rna_node_label_aggregator_output = self.rna_node_label_aggregator(
                    x=x,
                    edge_index=edge_index,
                    return_agg_weights=return_agg_weights)
            x_neighbors = rna_node_label_aggregator_output[0]
 
            # Retrieve rna node labels and only keep nodes in current node batch
            # and reconstructed features
            assert x.size(1) == self.n_output_genes_
            assert x_neighbors.size(1) == self.n_output_genes_
            output["node_labels"]["target_rna"] = x[batch_idx][
                :, self.features_idx_dict_["target_reconstructed_rna_idx"]]
            output["node_labels"]["source_rna"] = x_neighbors[batch_idx][
                :, self.features_idx_dict_["source_reconstructed_rna_idx"]]
            
            # Use observed library size as scaling factor for the negative
            # binomial means of the rna distribution
            target_rna_library_size = output["node_labels"]["target_rna"].sum(
                1).unsqueeze(1)
            source_rna_library_size = output["node_labels"]["source_rna"].sum(
                1).unsqueeze(1)
            self.target_rna_log_library_size = torch.log(target_rna_library_size)
            self.source_rna_log_library_size = torch.log(source_rna_library_size)

            # Get gene expression reconstruction distribution parameters for
            # reconstructed genes
            output["target_rna_nb_means"] = self.target_rna_decoder(
                z=z,
                log_library_size=self.target_rna_log_library_size,
                cat_covariates_embed=(
                    self.cat_covariates_embed[batch_idx] if
                    (self.cat_covariates_embed is not None) &
                    ("gene_expr_decoder" in
                     self.cat_covariates_embeds_injection_)
                     else None))[
                    :, self.features_idx_dict_["target_reconstructed_rna_idx"]]
            output["source_rna_nb_means"] = self.source_rna_decoder(
                z=z,
                log_library_size=self.source_rna_log_library_size,
                cat_covariates_embed=(
                self.cat_covariates_embed[batch_idx] if
                (self.cat_covariates_embed is not None) &
                ("gene_expr_decoder" in
                 self.cat_covariates_embeds_injection_)
                 else None))[
                    :, self.features_idx_dict_["source_reconstructed_rna_idx"]]
            
            if "atac" in self.modalities_:
                # Determine which features should be reconstructed based on
                # masks (if a feature is not connected to any node it should not
                # be reconstructed to not influence softmax activation outputs)
                if self.n_addon_gp_ > 0:
                    target_atac_decoder_static_mask = torch.cat(
                        (self.target_atac_decoder_mask,
                         self.target_atac_decoder_addon_mask), dim=0)
                    source_atac_decoder_static_mask = torch.cat(
                        (self.source_atac_decoder_mask,
                         self.source_atac_decoder_addon_mask), dim=0)
                else:
                    target_atac_decoder_static_mask = self.target_atac_decoder_mask
                    source_atac_decoder_static_mask = self.source_atac_decoder_mask

                self.target_n_gps_per_peak = (
                    target_atac_decoder_static_mask
                    * self.target_atac_dynamic_decoder_mask
                    ).sum(0)
                self.features_idx_dict_["target_reconstructed_atac_idx"] = (
                    torch.nonzero(self.target_n_gps_per_peak)).flatten().tolist()

                self.source_n_gps_per_peak = (
                    source_atac_decoder_static_mask
                    * self.source_atac_dynamic_decoder_mask
                    ).sum(0)
                self.features_idx_dict_["source_reconstructed_atac_idx"] = (
                    torch.nonzero(self.source_n_gps_per_peak)).flatten().tolist()

                self.target_atac_theta_reconstructed = self.target_atac_theta[
                    self.features_idx_dict_["target_reconstructed_atac_idx"]]
                self.source_atac_theta_reconstructed = self.source_atac_theta[
                    self.features_idx_dict_["source_reconstructed_atac_idx"]]

                # Compute aggregated neighborhood atac feature vector
                atac_node_label_aggregator_output = (
                    self.atac_node_label_aggregator(
                        x=x_atac,
                        edge_index=edge_index,
                        return_agg_weights=return_agg_weights))
                x_neighbors_atac = atac_node_label_aggregator_output[0]

                # Retrieve node labels and only keep nodes in current node batch
                # and reconstructed features
                assert x_atac.size(1) == self.n_output_peaks_
                assert x_neighbors_atac.size(1) == self.n_output_peaks_
                output["node_labels"]["target_atac"] = x_atac[batch_idx][
                    :, self.features_idx_dict_["target_reconstructed_atac_idx"]]  
                output["node_labels"]["source_atac"] = x_neighbors_atac[batch_idx][
                    :, self.features_idx_dict_["source_reconstructed_atac_idx"]]

                # Use observed library size as scaling factor for the negative
                # binomial means of the atac distribution
                target_atac_library_size = output["node_labels"][
                    "target_atac"].sum(1).unsqueeze(1)
                source_atac_library_size = output["node_labels"][
                    "source_atac"].sum(1).unsqueeze(1)
                self.target_atac_log_library_size = torch.log(
                    target_atac_library_size)
                self.source_atac_log_library_size = torch.log(
                    source_atac_library_size)
                
                if update_atac_dynamic_decoder_mask:
                    # Get atac dynamic decoder masks to turn off peaks that
                    # are mapped to only genes that are turned off
                    with torch.no_grad():
                        # Retrieve rna decoder gp weights
                        gp_weights = self.get_gp_weights(
                            only_masked_features=False)[0].detach().cpu()
                        
                        # Round to 4 decimals as genes are never completely
                        # turned off due to L1 being not differentiable at 0
                        gp_weights = torch.round(gp_weights, decimals=4)

                        # Get boolean mask of non zero target and source gene 
                        # weights
                        non_zero_gene_weights = torch.ne(
                                gp_weights, 
                                0) # dim: (2 x n_genes, n_gps)
                        non_zero_target_gene_weights = non_zero_gene_weights[
                            :self.n_output_genes_, :] # dim: (n_genes, n_gps)
                        non_zero_source_gene_weights = non_zero_gene_weights[
                            self.n_output_genes_:, :] # dim: (n_genes, n_gps)
                        
                        # Multiply boolean mask with gene peak mapping to remove
                        # peaks that are mapped to only turned off genes
                        target_atac_dynamic_decoder_mask = torch.mm(
                            non_zero_target_gene_weights.t().to(torch.float32), # dim: (n_gps,
                                                              #       n_genes)
                            self.gene_peaks_mask_.to(torch.float32)).to(torch.bool) # dim: (n_genes,
                                                   # n_peaks)
                            # dim: (n_gps, n_peaks)
                        source_atac_dynamic_decoder_mask = torch.mm(
                            non_zero_source_gene_weights.t().to(torch.float32),
                            self.gene_peaks_mask_.to(torch.float32)).to(torch.bool)
                        
                        # Create boolean mask of peaks (until here multiple
                        # active genes in a gp can be mapped to the same peak,
                        # resulting in values > 1.)
                        self.target_atac_dynamic_decoder_mask = (
                            self.target_atac_dynamic_decoder_mask & torch.ne(
                            target_atac_dynamic_decoder_mask, 
                            0)) # dim: (n_gps, n_peaks)
                        self.source_atac_dynamic_decoder_mask = (
                            self.source_atac_dynamic_decoder_mask & torch.ne(
                            source_atac_dynamic_decoder_mask, 
                            0))
                    
                # Get chromatin accessibility reconstruction distribution
                # parameters for reconstructed peaks
                output["target_atac_nb_means"] = self.target_atac_decoder(
                    z=z,
                    log_library_size=self.target_atac_log_library_size,
                    dynamic_mask=self.target_atac_dynamic_decoder_mask,
                    cat_covariates_embed=(
                        self.cat_covariates_embed[batch_idx] if
                        (self.cat_covariates_embed is not None) & 
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) else
                        None))[
                    :, self.features_idx_dict_["target_reconstructed_atac_idx"]]
                output["source_atac_nb_means"] = self.source_atac_decoder(
                    z=z,
                    log_library_size=self.source_atac_log_library_size,
                    dynamic_mask=self.source_atac_dynamic_decoder_mask,
                    cat_covariates_embed=(
                        self.cat_covariates_embed[batch_idx] if
                        (self.cat_covariates_embed is not None) &
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) else
                        None))[
                    :, self.features_idx_dict_["source_reconstructed_atac_idx"]]
        elif decoder == "graph":
            # Store edge labels in output for loss computation
            output["edge_recon_labels"] = data_batch.edge_label
                
            # For each categorical covariate, create a boolean tensor to
            # indicate for each sampled edge (negative & positive edges) whether
            # the edge nodes have the same category and store in a list
            if len(self.cat_covariates_cats_) > 0:
                output["edge_same_cat_covariates_cat"] = []
                for cat_covariate_idx in range(len(self.cat_covariates_cats_)):
                    edge_same_cat_covariate_cat = (
                        data_batch.cat_covariates_cats[
                            data_batch.edge_label_index[0],
                            cat_covariate_idx] ==
                        data_batch.cat_covariates_cats[
                            data_batch.edge_label_index[1],
                            cat_covariate_idx])
                    output["edge_same_cat_covariates_cat"].append(
                        edge_same_cat_covariate_cat)
                
                # Based on the categorical covariate and its possibility for
                # edges to exist for different categories (this might only be
                # the case for certain categorical covariates, others might only
                # use disconnected neighbor graphs for different categories),
                # create a boolean mask for edges whether they should be
                # included in the edge reconstruction loss and edge
                # reconstruction performance evaluation
                cat_covariates_cat_edge_incl = []
                for cat_covariate_no_edge, edge_same_cat_covariate_cat in zip(
                    self.cat_covariates_no_edges_,
                    output["edge_same_cat_covariates_cat"]):
                    if not cat_covariate_no_edge:
                        cat_covariates_cat_edge_incl.append(
                            torch.ones_like(edge_same_cat_covariate_cat,
                                            dtype=torch.bool))
                    else:
                        cat_covariates_cat_edge_incl.append(
                            edge_same_cat_covariate_cat)
                output["edge_incl"] = torch.all(
                    torch.stack(cat_covariates_cat_edge_incl),
                                dim=0)
            else:
                output["edge_same_cat_covariates_cat"] = None
                output["edge_incl"] = None

            # Use decoder to get the edge reconstruction logits
            output["edge_recon_logits"] = self.graph_decoder(z=z)
        return output


class CustomSpatialAnnTorchDataset(SpatialAnnTorchDataset):
    """
    Project-specific dataset wrapper for NicheCompass.

    This is the place to customize how RNA + ATAC are read/concatenated into the
    node feature matrix `self.x`, or how covariates / graph are prepared.

    By default this simply calls the upstream implementation.
    """

    def __init__(self,
                 adata: AnnData,
                 cat_covariates_label_encoders: List[dict],
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 edge_label_adj_key: str="edge_label_spatial_connectivities",
                 self_loops: bool=True,
                 cat_covariates_keys: Optional[str]=None):
        if counts_key is None:
            x_rna = adata.X
        else:
            x_rna = adata.layers[counts_key]

        # Store features in dense format
        if sp.issparse(x_rna): 
            self.x_rna = torch.tensor(x_rna.toarray())
        else:
            self.x_rna = torch.tensor(x_rna)

        # Concatenate ATAC feature vector in dense format if provided
        if adata_atac is not None:
            if sp.issparse(adata_atac.X): 
                self.x_atac = torch.tensor(adata_atac.X.toarray())
            else:
                self.x_atac = torch.tensor(adata_atac.X)

        # Store adjacency matrix in torch_sparse SparseTensor format
        if sp.issparse(adata.obsp[adj_key]):
            self.adj = sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        else:
            self.adj = sparse_mx_to_sparse_tensor(
                sp.csr_matrix(adata.obsp[adj_key]))
            
        # Store edge label adjacency matrix
        if edge_label_adj_key in adata.obsp:
            self.edge_label_adj = sp.csr_matrix(adata.obsp[edge_label_adj_key])
        else:
            self.edge_label_adj = None

        # Validate adjacency matrix symmetry
        if (self.adj.nnz() != self.adj.t().nnz()):
            raise ImportError("The input adjacency matrix has to be symmetric.")
        
        self.edge_index = self.adj.to_torch_sparse_coo_tensor()._indices()

        if self_loops:
            # Add self loops to account for autocrine communication
            # Remove self loops in case there are already before adding new ones
            self.edge_index, _ = remove_self_loops(self.edge_index)
            self.edge_index, _ = add_self_loops(self.edge_index,
                                                num_nodes=self.x.size(0))
            
        if cat_covariates_keys is not None:
            self.cat_covariates_cats = []
            for cat_covariate_key, cat_covariate_label_encoder in zip(
                cat_covariates_keys,
                cat_covariates_label_encoders):
                cat_covariate_cats = torch.tensor(
                    encode_labels(adata,
                                  cat_covariate_label_encoder,
                                  cat_covariate_key), dtype=torch.long)
                self.cat_covariates_cats.append(cat_covariate_cats)
            self.cat_covariates_cats = torch.stack(self.cat_covariates_cats,
                                                   dim=1)            

        self.n_node_features = self.x.size(1)
        self.size_factors = self.x.sum(1) # fix for ATAC case

    def __len__(self):
        """Return the number of observations stored in SpatialAnnTorchDataset"""
        return self.x_rna.size(0) + self.x_atac.size(0)


def prepare_data(adata: AnnData,
                 cat_covariates_label_encoders: List[dict],
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 cat_covariates_keys: Optional[List[str]]=None,
                 edge_val_ratio: float=0.1,
                 edge_test_ratio: float=0.,
                 node_val_ratio: float=0.1,
                 node_test_ratio: float=0.) -> dict:
    """
    Prepare data for model training including edge-level and node-level train, 
    validation, and test splits.

    Parameters
    ----------
    adata:
        AnnData object with counts stored in ´adata.layers[counts_key]´ or
        ´adata.X´ depending on ´counts_key´, and sparse adjacency matrix stored
        in ´adata.obsp[adj_key]´.
    adata_atac:
        Additional optional AnnData object with paired spatial ATAC data.
    cat_covariates_label_encoders:
        List of categorical covariates label encoders from the model (label
        encoding indeces need to be aligned with the ones from the model to get
        the correct categorical covariates embeddings).
    counts_key:
        Key under which the counts are stored in ´adata.layer´. If ´None´, uses
        ´adata.X´ as counts.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    cat_covariates_keys:
        Keys under which the categorical covariates are stored in ´adata.obs´.
    edge_val_ratio:
        Fraction of the data that is used as validation set on edge-level.
    edge_test_ratio:
        Fraction of the data that is used as test set on edge-level.
    node_val_ratio:
        Fraction of the data that is used as validation set on node-level.
    node_test_ratio:
        Fraction of the data that is used as test set on node-level.

    Returns
    ----------
    data_dict:
        Dictionary containing edge-level training, validation and test PyG 
        Data objects and node-level PyG Data object with split masks under keys 
        ´edge_train_data´, ´edge_val_data´, ´edge_test_data´, and 
        ´node_masked_data´ respectively. The edge-level PyG Data objects contain
        edges in the ´edge_label_index´ attribute and edge labels in the 
        ´edge_label´ attribute.
    """
    data_dict = {}
    dataset = CustomSpatialAnnTorchDataset(
        adata=adata,
        adata_atac=adata_atac,
        counts_key=counts_key,
        adj_key=adj_key,
        cat_covariates_keys=cat_covariates_keys,
        cat_covariates_label_encoders=cat_covariates_label_encoders)

    # PyG Data object (has 2 edge index pairs for one edge because of symmetry;
    # one edge index pair will be removed in the edge-level split).
    data_rna = Data(x=dataset.x_rna,
                edge_index=dataset.edge_index,
                edge_attr=dataset.edge_index.t()) # store index of edge nodes as
                                                # edge attribute for
                                                # aggregation weight retrieval
                                                # in mini batches
    data_atac = Data(x=dataset.x_atac,
                    edge_index=dataset.edge_index,
                    edge_attr=dataset.edge_index.t()) # store index of edge nodes as
                                                    # edge attribute for
                                                    # aggregation weight retrieval
                                                    # in mini batches

    # Concatenate rna and atac data
    data = data_rna.cat([data_atac])

    if cat_covariates_keys is not None:
        data.cat_covariates_cats = dataset.cat_covariates_cats

    # Edge-level split for edge reconstruction
    edge_train_data, edge_val_data, edge_test_data = edge_level_split(
        data=data,
        edge_label_adj=dataset.edge_label_adj,
        val_ratio=edge_val_ratio,
        test_ratio=edge_test_ratio)
    data_dict["edge_train_data"] = edge_train_data
    data_dict["edge_val_data"] = edge_val_data
    data_dict["edge_test_data"] = edge_test_data

    # Node-level split for gene expression reconstruction
    data_dict["node_masked_data"] = node_level_split_mask(
        data=data,
        val_ratio=node_val_ratio,
        test_ratio=node_test_ratio)
    return data_dict


__all__ = ["CustomVGPGAE", "CustomSpatialAnnTorchDataset"]
