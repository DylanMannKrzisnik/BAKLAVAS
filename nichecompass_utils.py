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
from nichecompass.models import NicheCompass

import torch
import scipy as sp
from anndata import AnnData
from torch_geometric.utils import add_self_loops, remove_self_loops
from nichecompass.data.utils import encode_labels, sparse_mx_to_sparse_tensor
from typing import List, Optional

# Isolate functions from dataprocessors to avoid circular imports
edge_level_split = dataprocessors.edge_level_split
node_level_split_mask = dataprocessors.node_level_split_mask

class CustomNicheCompass(NicheCompass):
    """
    Project-specific NicheCompass with custom behavior.
    """
    def __init__(self,
                 adata: AnnData,
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 gp_names_key: str="nichecompass_gp_names",
                 active_gp_names_key: str="nichecompass_active_gp_names",
                 gp_targets_mask_key: str="nichecompass_gp_targets",
                 gp_targets_categories_mask_key: str="nichecompass_gp_targets_categories",
                 targets_categories_label_encoder_key: str="nichecompass_targets_categories_label_encoder",
                 gp_sources_mask_key: str="nichecompass_gp_sources",
                 gp_sources_categories_mask_key: str="nichecompass_gp_sources_categories",
                 sources_categories_label_encoder_key: str="nichecompass_sources_categories_label_encoder",
                 ca_targets_mask_key: Optional[str]="nichecompass_ca_targets",
                 ca_sources_mask_key: Optional[str]="nichecompass_ca_sources",
                 latent_key: str="nichecompass_latent",
                 cat_covariates_embeds_keys: Optional[List[str]]=None,
                 cat_covariates_embeds_injection: Optional[List[
                     Literal["encoder",
                             "gene_expr_decoder",
                             "chrom_access_decoder"]]]=["gene_expr_decoder",
                                                        "chrom_access_decoder"],
                 cat_covariates_keys: Optional[List[str]]=None,
                 cat_covariates_no_edges: Optional[List[bool]]=None,
                 genes_idx_key: str="nichecompass_genes_idx",
                 target_genes_idx_key: str="nichecompass_target_genes_idx",
                 source_genes_idx_key: str="nichecompass_source_genes_idx",
                 peaks_idx_key: str="nichecompass_peaks_idx",
                 target_peaks_idx_key: str="nichecompass_target_peaks_idx",
                 source_peaks_idx_key: str="nichecompass_source_peaks_idx",
                 gene_peaks_mask_key: str="nichecompass_gene_peaks",
                 recon_adj_key: Optional[str]="nichecompass_recon_connectivities",
                 agg_weights_key: Optional[str]="nichecompass_agg_weights",
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 include_chrom_access_recon_loss: Optional[bool]=True,
                 include_cat_covariates_contrastive_loss: bool=False,
                 gene_expr_recon_dist: Literal["nb"]="nb",
                 log_variational: bool=True,
                 node_label_method: Literal[
                    "one-hop-sum",
                    "one-hop-norm",
                    "one-hop-attention"]="one-hop-norm",
                 active_gp_thresh_ratio: float=0.01,
                 active_gp_type: Literal["mixed", "separate"]="separate",
                 n_fc_layers_encoder: int=1,
                 n_layers_encoder: int=1,
                 n_hidden_encoder: Optional[int]=None,
                 conv_layer_encoder: Literal["gcnconv", "gatv2conv"]="gatv2conv",
                 encoder_n_attention_heads: Optional[int]=4,
                 encoder_use_bn: bool=False,
                 dropout_rate_encoder: float=0.,
                 dropout_rate_graph_decoder: float=0.,
                 cat_covariates_cats: Optional[List[List]]=None,
                 n_addon_gp: int=100,
                 cat_covariates_embeds_nums: Optional[List[int]]=None,
                 include_edge_kl_loss: bool=True,
                 use_cuda_if_available: bool=True,
                 seed: int=0,
                 **kwargs):
        self.adata = adata
        self.adata_atac = adata_atac
        self.counts_key_ = counts_key
        self.adj_key_ = adj_key
        self.gp_names_key_ = gp_names_key
        self.active_gp_names_key_ = active_gp_names_key
        self.gp_targets_mask_key_ = gp_targets_mask_key
        self.gp_targets_categories_mask_key_ = gp_targets_categories_mask_key
        self.targets_categories_label_encoder_key_ = (
            targets_categories_label_encoder_key)
        self.gp_sources_mask_key_ = gp_sources_mask_key
        self.gp_sources_categories_mask_key_ = gp_sources_categories_mask_key
        self.sources_categories_label_encoder_key_ = (
            sources_categories_label_encoder_key)
        self.ca_targets_mask_key_ = ca_targets_mask_key
        self.ca_sources_mask_key_ = ca_sources_mask_key
        self.latent_key_ = latent_key
        self.cat_covariates_embeds_keys_ = cat_covariates_embeds_keys
        self.cat_covariates_embeds_injection_ = cat_covariates_embeds_injection
        self.cat_covariates_keys_ = cat_covariates_keys
        self.cat_covariates_embeds_keys_ = cat_covariates_embeds_keys
        self.genes_idx_key_ = genes_idx_key
        self.target_genes_idx_key_ = target_genes_idx_key
        self.source_genes_idx_key_ = source_genes_idx_key
        self.peaks_idx_key_ = peaks_idx_key
        self.target_peaks_idx_key_ = target_peaks_idx_key
        self.source_peaks_idx_key_ = source_peaks_idx_key
        self.gene_peaks_mask_key_ = gene_peaks_mask_key
        self.recon_adj_key_ = recon_adj_key
        self.agg_weights_key_ = agg_weights_key
        self.include_edge_recon_loss_ = include_edge_recon_loss
        self.include_gene_expr_recon_loss_ = include_gene_expr_recon_loss
        self.include_chrom_access_recon_loss_ = include_chrom_access_recon_loss
        self.include_cat_covariates_contrastive_loss_ = (
            include_cat_covariates_contrastive_loss)
        self.gene_expr_recon_dist_ = gene_expr_recon_dist
        self.log_variational_ = log_variational
        self.node_label_method_ = node_label_method
        self.active_gp_thresh_ratio_ = active_gp_thresh_ratio
        self.active_gp_type_ = active_gp_type
        self.include_edge_kl_loss_ = include_edge_kl_loss
        self.seed_ = seed

        # Set seed for reproducibility
        np.random.seed(self.seed_)
        if use_cuda_if_available & torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_)
            torch.manual_seed(self.seed_)
        else:
            torch.manual_seed(self.seed_)

        # Retrieve gene program masks
        if gp_targets_mask_key in adata.varm:
            # NOTE: dtype can be changed to bool and should be able to handle sparse
            # mask
            self.gp_targets_mask_ = torch.tensor(
                adata.varm[gp_targets_mask_key].T,
                dtype=torch.bool)
        else:
            raise ValueError("Please specify an adequate ´gp_targets_mask_key´ "
                             "for your adata object. The targets mask needs to "
                             "be stored in ´adata.varm[gp_targets_mask_key]´. "
                             " If you do not want to mask gene expression "
                             "reconstruction, you can create a mask of 1s that"
                             " allows all gene program latent nodes to "
                             "reconstruct all genes.")

        if gp_sources_mask_key in adata.varm:
            # NOTE: dtype can be changed to bool and should be able to handle
            # sparse mask
            self.gp_sources_mask_ = torch.tensor(
                adata.varm[gp_sources_mask_key].T,
                dtype=torch.bool)
                                           
        else:
            raise ValueError("Please specify an adequate "
                             "´gp_sources_mask_key´ for your adata object. "
                             "The sources mask needs to be stored in "
                             "´adata.varm[gp_sources_mask_key]´. If you do "
                             "not want to mask gene expression "
                             "reconstruction, you can create a mask of 1s "
                             " that allows all gene program latent nodes to"
                             " reconstruct all genes.")
            
        # Determine features scale factors
        self.features_scale_factors_ = torch.concat(
            (torch.tensor(self.adata.X.sum(0))[0],
             torch.tensor(self.adata.X.sum(0))[0]))
    
        # Retrieve chromatin accessibility masks
        if adata_atac is None:
            self.ca_targets_mask_ = None
            self.ca_sources_mask_ = None
            gene_peaks_mask = None
        else:
            gene_peaks_mask = adata.varm[gene_peaks_mask_key].tocoo()
            gene_peaks_mask = torch.sparse_coo_tensor(
                indices=[gene_peaks_mask.row, gene_peaks_mask.col],
                values=gene_peaks_mask.data,
                size=gene_peaks_mask.shape,
                dtype=torch.bool) # bool does not work with torch.mm
            if ca_targets_mask_key in adata_atac.varm:
                ca_targets_mask = adata_atac.varm[ca_targets_mask_key].T.tocoo()
            else:
                raise ValueError("Please specify an adequate "
                                 "´ca_targets_mask_key´ for your adata_atac "
                                 "object. The targets mask needs to be stored "
                                 "in ´adata_atac.varm[ca_targets_mask_key]´. If"
                                 " you do not want to mask chromatin "
                                 " accessibility reconstruction, you can create"
                                 " a mask of 1s that allows all gene program "
                                 "latent nodes to reconstruct all peaks.")
            self.ca_targets_mask_ = torch.sparse_coo_tensor(
                indices=[ca_targets_mask.row, ca_targets_mask.col],
                values=ca_targets_mask.data,
                size=ca_targets_mask.shape,
                dtype=torch.bool).to_dense() # for now
            if ca_sources_mask_key in adata_atac.varm:
                ca_sources_mask = adata_atac.varm[
                    ca_sources_mask_key].T.tocoo()
                self.ca_sources_mask_ = torch.sparse_coo_tensor(
                    indices=[ca_sources_mask.row, ca_sources_mask.col],
                    values=ca_sources_mask.data,
                    size=ca_sources_mask.shape,
                    dtype=torch.bool).to_dense() # for now
            else:
                raise ValueError("Please specify an adequate "
                                "´ca_sources_mask_key´ for your adata_atac "
                                "object. The sources mask needs to be "
                                "stored in "
                                "´adata_atac.varm[ca_sources_mask_key]´. If"
                                "you do not want to mask chromatin "
                                " accessibility reconstruction, you can "
                                "create a mask of 1s that allows all gene "
                                "program latent nodes to reconstruct all "
                                "peaks.")

        # Retrieve index of genes in gp mask and index of genes not in gp mask
        self.features_idx_dict_ = {}
        self.features_idx_dict_["masked_rna_idx"] = adata.uns[
            genes_idx_key]
        self.features_idx_dict_["unmasked_rna_idx"] = [
            i for i in range(len(adata.var_names))
            if i not in self.features_idx_dict_["masked_rna_idx"]]
        self.features_idx_dict_["target_masked_rna_idx"] = list(
            adata.uns[target_genes_idx_key])
        self.features_idx_dict_["target_unmasked_rna_idx"] = [
            i for i in range(len(adata.var_names))
            if i not in self.features_idx_dict_["target_masked_rna_idx"]]
        self.features_idx_dict_["source_masked_rna_idx"] = list(
            adata.uns[source_genes_idx_key])
        self.features_idx_dict_["source_unmasked_rna_idx"] = [
            i for i in range(len(adata.var_names))
            if i not in self.features_idx_dict_["source_masked_rna_idx"]]
        
        # Retrieve index of peaks in ca mask and index of peaks not in ca mask
        if adata_atac is not None:
            self.peaks_idx_ = adata_atac.uns[peaks_idx_key]
            self.target_peaks_idx_ = adata_atac.uns[target_peaks_idx_key]
            self.source_peaks_idx_ = adata_atac.uns[source_peaks_idx_key]
            
            self.features_idx_dict_["masked_atac_idx"] = adata_atac.uns[
                peaks_idx_key]
            self.features_idx_dict_["unmasked_atac_idx"] = [
                i for i in range(len(adata_atac.var_names))
                if i not in self.features_idx_dict_["masked_atac_idx"]]
            self.features_idx_dict_["target_masked_atac_idx"] = list(
                adata_atac.uns[target_peaks_idx_key])
            self.features_idx_dict_["target_unmasked_atac_idx"] = [
                i for i in range(len(adata_atac.var_names))
                if i not in self.features_idx_dict_["target_masked_atac_idx"]]
            self.features_idx_dict_["source_masked_atac_idx"] = list(
                adata_atac.uns[source_peaks_idx_key])
            self.features_idx_dict_["source_unmasked_atac_idx"] = [
                i for i in range(len(adata_atac.var_names))
                if i not in self.features_idx_dict_["source_masked_atac_idx"]]

        # Determine VGPGAE inputs
        self.n_input_ = adata.n_vars
        self.n_output_genes_ = adata.n_vars
        if adata_atac is not None:
            self.modalities_ = ["rna", "atac"]
            if not np.all(adata.obs.index == adata_atac.obs.index):
                raise ValueError("Please make sure that 'adata' and "
                                 "'adata_atac' contain the same observations in"
                                 " the same order.")
            # Peaks are concatenated to genes in input
            self.n_input_ += adata_atac.n_vars
            self.n_output_peaks_ = adata_atac.n_vars
        else:
            self.modalities_ = ["rna"]
            self.n_output_peaks_ = 0
        self.n_fc_layers_encoder_ = n_fc_layers_encoder
        self.n_layers_encoder_ = n_layers_encoder
        self.conv_layer_encoder_ = conv_layer_encoder
        if conv_layer_encoder == "gatv2conv":
            self.encoder_n_attention_heads_ = encoder_n_attention_heads
        else:
            self.encoder_n_attention_heads_ = 0
        self.encoder_use_bn_ = encoder_use_bn
        self.dropout_rate_encoder_ = dropout_rate_encoder
        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder
        self.n_prior_gp_ = len(self.gp_targets_mask_)
        self.n_addon_gp_ = n_addon_gp
        
        if n_addon_gp > 0:
            # Add add-on gps to adata
            gp_list = list(self.adata.uns[self.gp_names_key_])
            for i in range(n_addon_gp):
                if f"Add-on_{i}_GP" not in gp_list:
                    gp_list.append(f"Add-on_{i}_GP")
            self.adata.uns[self.gp_names_key_] = np.array(gp_list)
        else:
            # Remove add-on gps from adata
            for gp_name in list(adata.uns[gp_names_key]):
                if "Add-on" in gp_name:
                    self.adata.uns[gp_names_key] = np.delete(
                        self.adata.uns[gp_names_key],
                        list(self.adata.uns[gp_names_key]).index(gp_name))

        # Retrieve categorical covariates categories
        if cat_covariates_cats is None:
            if cat_covariates_keys is not None:
                self.cat_covariates_cats_ = [
                    adata.obs[cat_covariate_key].unique().tolist() 
                    for cat_covariate_key in cat_covariates_keys]
            else:
                self.cat_covariates_cats_ = []
        else:
            self.cat_covariates_cats_ = cat_covariates_cats
        
        # Define dimensionality of categorical covariates embeddings as
        # number of categories of each categorical covariate respectively
        # if not provided explicitly
        if cat_covariates_embeds_nums is None:
            cat_covariates_embeds_nums = []
            for cat_covariate_cats in self.cat_covariates_cats_:
                cat_covariates_embeds_nums.append(len(cat_covariate_cats))
        self.cat_covariates_embeds_nums_ = cat_covariates_embeds_nums

        # Determine dimensionality of hidden encoder layer if not provided
        if n_hidden_encoder is None:
            if len(adata.var) > (self.n_prior_gp_ + self.n_addon_gp_):
                n_hidden_encoder = (self.n_prior_gp_ + self.n_addon_gp_)
            else:
                n_hidden_encoder = len(adata.var)
        self.n_hidden_encoder_ = n_hidden_encoder
            
        # Define categorical covariates no edges as all 'True' if not
        # explicitly provided, so that they are excluded from the edge
        # reconstruction loss
        if ((cat_covariates_no_edges is None) &
            (len(self.cat_covariates_cats_) > 0)):
            self.cat_covariates_no_edges_ = (
                [True] * len(self.cat_covariates_cats_))
        else:
            self.cat_covariates_no_edges_ = cat_covariates_no_edges
        
        # Validate counts layer key and counts values
        if counts_key is not None and counts_key not in adata.layers:
            raise ValueError("Please specify an adequate ´counts_key´. By "
                             "default the counts are assumed to be stored in "
                             "data.layers['counts'].")
        if include_gene_expr_recon_loss and log_variational:
            if counts_key is None:
                x = adata.X
            else:
                x = adata.layers[counts_key]
            if (x < 0).sum() > 0:
                raise ValueError("Please make sure that "
                                 "´adata.layers[counts_key]´ contains the"
                                 " raw counts (not log library size "
                                 "normalized) if ´include_gene_expr_recon_loss´"
                                 " is ´True´ and ´log_variational´ is ´True´. "
                                 "If you want to use log library size "
                                 " normalized counts, make sure that "
                                 "´log_variational´ is ´False´.")

        # Validate adjacency key
        if adj_key not in adata.obsp:
            raise ValueError("Please specify an adequate ´adj_key´. "
                             "By default the adjacency matrix is assumed to be "
                             "stored in adata.obsm['spatial_connectivities'].")

        # Validate gp key
        if gp_names_key not in adata.uns:
            raise ValueError("Please specify an adequate ´gp_names_key´. "
                             "By default the gene program names are assumed to "
                             "be stored in adata.uns['nichecompass_gp_names'].")

        # Validate categorical covariates keys
        if cat_covariates_keys is not None:
            for cat_covariate_key in cat_covariates_keys:
                if cat_covariate_key not in adata.obs:
                    raise ValueError(
                        "Please specify adequate ´cat_covariates_keys´. "
                        f"The key {cat_covariate_key} was not found in adata.")
        
        # Initialize model with Variational Gene Program Graph Autoencoder 
        # neural network module
        self.model = CustomVGPGAE(
            n_input=self.n_input_,
            n_fc_layers_encoder=self.n_fc_layers_encoder_,
            n_layers_encoder=self.n_layers_encoder_,
            n_hidden_encoder=self.n_hidden_encoder_,
            n_prior_gp=self.n_prior_gp_,
            n_addon_gp=self.n_addon_gp_,
            cat_covariates_embeds_nums=self.cat_covariates_embeds_nums_,
            n_output_genes=self.n_output_genes_,
            n_output_peaks=self.n_output_peaks_,
            target_rna_decoder_mask=self.gp_targets_mask_,
            source_rna_decoder_mask=self.gp_sources_mask_,
            target_atac_decoder_mask=self.ca_targets_mask_,
            source_atac_decoder_mask=self.ca_sources_mask_,
            features_idx_dict=self.features_idx_dict_,
            features_scale_factors=self.features_scale_factors_,
            gene_peaks_mask=gene_peaks_mask,
            cat_covariates_cats=self.cat_covariates_cats_,
            cat_covariates_no_edges=self.cat_covariates_no_edges_,
            conv_layer_encoder=self.conv_layer_encoder_,
            encoder_n_attention_heads=self.encoder_n_attention_heads_,
            encoder_use_bn=self.encoder_use_bn_,
            dropout_rate_encoder=self.dropout_rate_encoder_,
            dropout_rate_graph_decoder=self.dropout_rate_graph_decoder_,
            include_edge_recon_loss=self.include_edge_recon_loss_,
            include_gene_expr_recon_loss=self.include_gene_expr_recon_loss_,
            include_chrom_access_recon_loss=self.include_chrom_access_recon_loss_,
            include_cat_covariates_contrastive_loss=self.include_cat_covariates_contrastive_loss_,
            rna_recon_loss=self.gene_expr_recon_dist_,
            node_label_method=self.node_label_method_,
            active_gp_thresh_ratio=self.active_gp_thresh_ratio_,
            active_gp_type=self.active_gp_type_,
            log_variational=self.log_variational_,
            cat_covariates_embeds_injection=self.cat_covariates_embeds_injection_,
            include_edge_kl_loss=self.include_edge_kl_loss_)

        self.is_trained_ = False

        # Store init params for saving and loading
        self.init_params_ = self._get_init_params(locals())

class CustomVGPGAE(VGPGAE):
    """
    Project-specific VGPGAE with custom behavior.

    Override methods as needed; this is a safe default that calls the parent
    implementation while giving you a single place to modify logic.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    Project-specific prepare_data function imitating nichecompass.data.dataprocessors.prepare_data().
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
