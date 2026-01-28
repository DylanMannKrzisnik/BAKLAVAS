"""
Utilities for customizing NicheCompass components.

This module is intended to host project-specific subclasses or helpers that
override behavior from the NicheCompass package (e.g., custom VGPGAE forward).
"""

from __future__ import annotations

from typing import List, Literal, Optional

import math
import time
import warnings
from collections import defaultdict

import mlflow
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from anndata import AnnData
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops

from nichecompass.data import (SpatialAnnTorchDataset,
                               dataprocessors,
                               initialize_dataloaders)
from nichecompass.data.utils import encode_labels, sparse_mx_to_sparse_tensor
from nichecompass.models import NicheCompass
from nichecompass.modules import VGPGAE
from nichecompass.nn import Encoder
from nichecompass.train import Trainer
from nichecompass.train.metrics import eval_metrics
from nichecompass.train.utils import _cycle_iterable, print_progress

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
                 multimodal_embedding_size: int=128,
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
            #if not np.all(adata.obs.index == adata_atac.obs.index):
            #    raise ValueError("Please make sure that 'adata' and "
            #                     "'adata_atac' contain the same observations in"
            #                     " the same order.")
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
        self.multimodal_embedding_size_ = multimodal_embedding_size
        
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
            multimodal_embedding_size=self.multimodal_embedding_size_,
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

    @classmethod
    def load(cls,
             dir_path: str,
             adata: Optional[AnnData]=None,
             adata_atac: Optional[AnnData]=None,
             adata_file_name: str="adata.h5ad",
             adata_atac_file_name: Optional[str]="adata_atac.h5ad",
             use_cuda: bool=False,
             n_addon_gps: int=0,
             gp_names_key: Optional[str]=None,
             genes_idx_key: Optional[str]=None,
             unfreeze_all_weights: bool=False,
             unfreeze_addon_gp_weights: bool=False,
             unfreeze_cat_covariates_embedder_weights: bool=False
             ) -> torch.nn.Module:
        """
        Load a saved CustomNicheCompass model with ATAC defaults.
        """
        return super().load(
            dir_path=dir_path,
            adata=adata,
            adata_atac=adata_atac,
            adata_file_name=adata_file_name,
            adata_atac_file_name=adata_atac_file_name,
            use_cuda=use_cuda,
            n_addon_gps=n_addon_gps,
            gp_names_key=gp_names_key,
            genes_idx_key=genes_idx_key,
            unfreeze_all_weights=unfreeze_all_weights,
            unfreeze_addon_gp_weights=unfreeze_addon_gp_weights,
            unfreeze_cat_covariates_embedder_weights=(
                unfreeze_cat_covariates_embedder_weights))

    def train(self,
              n_epochs: int=100,
              n_epochs_all_gps: int=25,
              n_epochs_no_edge_recon: int=0,
              n_epochs_no_cat_covariates_contrastive: int=5,
              lr: float=0.001,
              weight_decay: float=0.,
              lambda_edge_recon: Optional[float]=500000.,
              lambda_gene_expr_recon: float=300.,
              lambda_chrom_access_recon: float=100.,
              lambda_cat_covariates_contrastive: float=0.,
              lambda_multimodal_contrastive_loss: float=0.,
              multimodal_temperature: float=1.0,
              multimodal_contrastive_anneal: bool=False,
              contrastive_logits_pos_ratio: float=0.,
              contrastive_logits_neg_ratio: float=0.,
              lambda_group_lasso: float=0.,
              lambda_l1_masked: float=0.,
              l1_targets_categories: Optional[list]=["target_gene"],
              l1_sources_categories: Optional[list]=None,
              lambda_l1_addon: float=30.,
              edge_val_ratio: float=0.1,
              node_val_ratio: float=0.1,
              edge_batch_size: int=256,
              node_batch_size: Optional[int]=None,
              paired_data: bool=True,
              mlflow_experiment_id: Optional[str]=None,
              retrieve_cat_covariates_embeds: bool=False,
              retrieve_recon_edge_probs: bool=False,
              retrieve_agg_weights: bool=False,
              use_cuda_if_available: bool=True,
              n_sampled_neighbors: int=-1,
              latent_dtype: type=np.float64,
              **trainer_kwargs):
        """
        Train the CustomNicheCompass model using CustomTrainer.
        """
        self.trainer = CustomTrainer(
            adata=self.adata,
            adata_atac=self.adata_atac,
            model=self.model,
            counts_key=self.counts_key_,
            adj_key=self.adj_key_,
            gp_targets_mask_key=self.gp_targets_mask_key_,
            gp_sources_mask_key=self.gp_sources_mask_key_,
            cat_covariates_keys=self.cat_covariates_keys_,
            edge_val_ratio=edge_val_ratio,
            node_val_ratio=node_val_ratio,
            edge_batch_size=edge_batch_size,
            node_batch_size=node_batch_size,
            paired_data=paired_data,
            use_cuda_if_available=use_cuda_if_available,
            n_sampled_neighbors=n_sampled_neighbors,
            latent_dtype=latent_dtype,
            **trainer_kwargs)
        
        if lambda_l1_masked > 0.:
            # Create mask for l1 regularization loss
            if l1_targets_categories is None:
                l1_targets_categories_encoded = list(self.adata.uns[
                    self.targets_categories_label_encoder_key_].values())
            else:
                l1_targets_categories_encoded = [
                    self.adata.uns[
                        self.targets_categories_label_encoder_key_][category]
                    for category in l1_targets_categories if category in
                    self.adata.uns[self.targets_categories_label_encoder_key_]]
            if l1_sources_categories is None:
                l1_sources_categories_encoded = list(self.adata.uns[
                    self.sources_categories_label_encoder_key_].values())
            else:
                l1_sources_categories_encoded = [
                    self.adata.uns[
                        self.sources_categories_label_encoder_key_][category]
                    for category in l1_sources_categories if category in
                    self.adata.uns[self.sources_categories_label_encoder_key_]]
            l1_targets_mask = torch.from_numpy(np.isin(
                self.adata.varm[self.gp_targets_categories_mask_key_],
                l1_targets_categories_encoded))
            l1_sources_mask = torch.from_numpy(np.isin(
                self.adata.varm[self.gp_sources_categories_mask_key_],
                l1_sources_categories_encoded))
        else:
            l1_targets_mask = None
            l1_sources_mask = None

        self.trainer.train(
            n_epochs=n_epochs,
            n_epochs_no_edge_recon=n_epochs_no_edge_recon,
            n_epochs_no_cat_covariates_contrastive=n_epochs_no_cat_covariates_contrastive,
            n_epochs_all_gps=n_epochs_all_gps,
            lr=lr,
            weight_decay=weight_decay,
            lambda_edge_recon=lambda_edge_recon,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_chrom_access_recon=lambda_chrom_access_recon,
            lambda_cat_covariates_contrastive=lambda_cat_covariates_contrastive,
            lambda_multimodal_contrastive_loss=lambda_multimodal_contrastive_loss,
            multimodal_temperature=multimodal_temperature,
            multimodal_contrastive_anneal=multimodal_contrastive_anneal,
            contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
            contrastive_logits_neg_ratio=contrastive_logits_neg_ratio,
            lambda_group_lasso=lambda_group_lasso,
            lambda_l1_masked=lambda_l1_masked,
            l1_targets_mask=l1_targets_mask,
            l1_sources_mask=l1_sources_mask,
            lambda_l1_addon=lambda_l1_addon,
            mlflow_experiment_id=mlflow_experiment_id)
        
        self.node_batch_size_ = self.trainer.node_batch_size_
        
        self.is_trained_ = True
        self.model.eval()

        self.adata.obsm[self.latent_key_], _ = self.get_latent_representation(
           adata=self.adata,
           counts_key=self.counts_key_,
           adj_key=self.adj_key_,
           cat_covariates_keys=self.cat_covariates_keys_,
           only_active_gps=True,
           return_mu_std=True,
           node_batch_size=self.node_batch_size_,
           dtype=latent_dtype)

        self.adata.uns[self.active_gp_names_key_] = self.get_active_gps()

        if ((len(self.cat_covariates_cats_) > 0) &
            retrieve_cat_covariates_embeds):
            for cat_covariates_embed_key, cat_covariate_embed in zip(
                self.cat_covariates_embeds_keys_,
                self.get_cat_covariates_embeddings()):
                self.adata.uns[cat_covariates_embed_key] = cat_covariate_embed

        if retrieve_recon_edge_probs:
            self.adata.obsp[self.recon_adj_key_] = self.get_recon_edge_probs()

        if retrieve_agg_weights:
            self.adata.obsp[self.agg_weights_key_] = (
                self.get_neighbor_importances(
                    node_batch_size=self.node_batch_size_))

        if mlflow_experiment_id is not None:
            mlflow.log_metric("n_active_gps",
                              len(self.adata.uns[self.active_gp_names_key_]))

    def get_latent_representation(
            self,
            adata: Optional[AnnData]=None,
            adata_atac: Optional[AnnData]=None,
            counts_key: Optional[str]="counts",
            adj_key: str="spatial_connectivities",
            cat_covariates_keys: Optional[List[str]]=None,
            paired_data: bool=True,
            only_active_gps: bool=True,
            return_mu_std: bool=False,
            node_batch_size: int=64,
            dtype: type=np.float64,
            ) -> np.ndarray:
        """
        Get the latent representation / gene program scores from a trained model.
        """
        self._check_if_trained(warn=False)

        device = next(self.model.parameters()).device

        if adata is None:
            adata = self.adata
        if (adata_atac is None) & hasattr(self, "adata_atac"):
            adata_atac = self.adata_atac

        # Create single dataloader containing entire dataset
        data_dict = prepare_data(
            adata=adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=adata_atac,
            counts_key=counts_key,
            adj_key=adj_key,
            cat_covariates_keys=cat_covariates_keys,
            paired_data=paired_data,
            edge_val_ratio=0.,
            edge_test_ratio=0.,
            node_val_ratio=0.,
            node_test_ratio=0.)
        node_masked_data = data_dict["node_masked_data"]
        loader_dict = initialize_dataloaders(
            node_masked_data=node_masked_data,
            edge_train_data=None,
            edge_val_data=None,
            edge_batch_size=None,
            node_batch_size=node_batch_size,
            shuffle=False)
        node_loader = loader_dict["node_train_loader"]

        # Get number of gene programs
        if only_active_gps:
            n_gps = self.get_active_gps().shape[0]
        else:
            n_gps = (self.n_prior_gp_ + self.n_addon_gp_ )

        n_obs = node_masked_data.num_nodes
        if return_mu_std:
            mu = np.empty(shape=(n_obs, n_gps), dtype=dtype)
            std = np.empty(shape=(n_obs, n_gps), dtype=dtype)
        else:
            z = np.empty(shape=(n_obs, n_gps), dtype=dtype)

        # Get latent representation for each batch of the dataloader and put it
        # into latent vectors
        # Get model dtype to ensure consistency
        model_dtype = next(self.model.parameters()).dtype
        for i, node_batch in enumerate(node_loader):
            n_obs_before_batch = i * node_batch_size
            n_obs_after_batch = n_obs_before_batch + node_batch.batch_size
            node_batch = node_batch.to(device)
            # Ensure node_batch.x has the same dtype as the model
            if node_batch.x.dtype != model_dtype:
                node_batch.x = node_batch.x.to(model_dtype)
            if return_mu_std:
                mu_batch, std_batch = self.model.get_latent_representation(
                    node_batch=node_batch,
                    only_active_gps=only_active_gps,
                    return_mu_std=True)
                mu[n_obs_before_batch:n_obs_after_batch, :] = (
                    mu_batch.detach().cpu().numpy())
                std[n_obs_before_batch:n_obs_after_batch, :] = (
                    std_batch.detach().cpu().numpy())
            else:
                z_batch = self.model.get_latent_representation(
                    node_batch=node_batch,
                    only_active_gps=only_active_gps,
                    return_mu_std=False)
                z[n_obs_before_batch:n_obs_after_batch, :] = (
                    z_batch.detach().cpu().numpy())
        if return_mu_std:
            return mu, std
        else:
            return z

    def get_omics_decoder_outputs(
            self,
            adata: Optional[AnnData]=None,
            adata_atac: Optional[AnnData]=None,
            paired_data: bool=True,
            only_active_gps: bool=True,
            node_batch_size: int=64,
            ) -> dict:
        """
        Get the omics decoder outputs.
        """
        self._check_if_trained(warn=False)

        device = next(self.model.parameters()).device

        if adata is None:
            adata = self.adata
        if (adata_atac is None) & hasattr(self, "adata_atac"):
            adata_atac = self.adata_atac

        # Create single dataloader containing entire dataset
        data_dict = prepare_data(
            adata=adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=adata_atac,
            counts_key=self.counts_key_,
            adj_key=self.adj_key_,
            cat_covariates_keys=self.cat_covariates_keys_,
            paired_data=paired_data,
            edge_val_ratio=0.,
            edge_test_ratio=0.,
            node_val_ratio=0.,
            node_test_ratio=0.)
        node_masked_data = data_dict["node_masked_data"]
        loader_dict = initialize_dataloaders(
            node_masked_data=node_masked_data,
            edge_train_data=None,
            edge_val_data=None,
            edge_batch_size=None,
            node_batch_size=node_batch_size,
            shuffle=False)
        node_loader = loader_dict["node_train_loader"]

        n_obs = node_masked_data.num_nodes
        output = {}
        output["target_rna_nb_means"] = np.empty(
            shape=(n_obs, self.n_output_genes_))
        output["source_rna_nb_means"] = np.empty(
            shape=(n_obs, self.n_output_genes_))
        if "atac" in self.modalities_:
            output["target_atac_nb_means"] = np.empty(
                shape=(n_obs, self.n_output_peaks_))
            output["source_atac_nb_means"] = np.empty(
                shape=(n_obs, self.n_output_peaks_))

        # Get latent representation for each batch of the dataloader and put it
        # into latent vectors
        # Get model dtype to ensure consistency
        model_dtype = next(self.model.parameters()).dtype
        for i, node_batch in enumerate(node_loader):
            n_obs_before_batch = i * node_batch_size
            n_obs_after_batch = n_obs_before_batch + node_batch.batch_size
            node_batch = node_batch.to(device)
            # Ensure node_batch.x has the same dtype as the model
            if node_batch.x.dtype != model_dtype:
                node_batch.x = node_batch.x.to(model_dtype)
            output_batch = self.model.get_omics_decoder_outputs(
                node_batch=node_batch,
                only_active_gps=only_active_gps)
            output["target_rna_nb_means"][
                n_obs_before_batch:n_obs_after_batch, :] = (
                    output_batch["target_rna_nb_means"].detach().cpu().numpy())
            output["source_rna_nb_means"][
                n_obs_before_batch:n_obs_after_batch, :] = (
                    output_batch["source_rna_nb_means"].detach().cpu().numpy())
            if "atac" in self.modalities_:
                output["target_atac_nb_means"][
                    n_obs_before_batch:n_obs_after_batch, :] = (
                        output_batch["target_atac_nb_means"].detach()
                        .cpu().numpy())
                output["source_atac_nb_means"][
                    n_obs_before_batch:n_obs_after_batch, :] = (
                        output_batch["source_atac_nb_means"].detach()
                        .cpu().numpy())
        return output

class CustomTrainer(Trainer):
    """
    Trainer that uses the project-specific prepare_data implementation.
    """

    def __init__(self, *args, paired_data: bool=True, **kwargs):
        super().__init__(*args, **kwargs)

        data_dict = prepare_data(
            adata=self.adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=self.adata_atac,
            counts_key=self.counts_key,
            adj_key=self.adj_key,
            cat_covariates_keys=self.cat_covariates_keys,
            paired_data=paired_data,
            edge_val_ratio=self.edge_val_ratio_,
            edge_test_ratio=0.,
            node_val_ratio=self.node_val_ratio_,
            node_test_ratio=0.)

        self.node_masked_data = data_dict["node_masked_data"]
        self.edge_train_data = data_dict["edge_train_data"]
        self.edge_val_data = data_dict["edge_val_data"]
        self.n_nodes_train = self.node_masked_data.train_mask.sum().item()
        self.n_nodes_val = self.node_masked_data.val_mask.sum().item()
        self.n_edges_train = self.edge_train_data.edge_label_index.size(1)
        self.n_edges_val = self.edge_val_data.edge_label_index.size(1)

        if self.node_batch_size_ is None:
            self.node_batch_size_ = int(self.edge_batch_size_ / math.floor(
                self.n_edges_train / self.n_nodes_train))

        loader_dict = initialize_dataloaders(
            node_masked_data=self.node_masked_data,
            edge_train_data=self.edge_train_data,
            edge_val_data=self.edge_val_data,
            edge_batch_size=self.edge_batch_size_,
            node_batch_size=self.node_batch_size_,
            n_direct_neighbors=self.n_sampled_neighbors_,
            n_hops=self.loaders_n_hops_,
            edges_directed=False,
            neg_edge_sampling_ratio=1.)
        self.edge_train_loader = loader_dict["edge_train_loader"]
        self.edge_val_loader = loader_dict.pop("edge_val_loader", None)
        self.node_train_loader = loader_dict["node_train_loader"]
        self.node_val_loader = loader_dict.pop("node_val_loader", None)

    def _get_multimodal_contrastive_weight(self, increasing: bool=True) -> float:
        if (not self.multimodal_contrastive_anneal_) or self.n_epochs_ <= 1:
            return self.lambda_multimodal_contrastive_loss_
        progress = self.epoch / max(1, self.n_epochs_ - 1)
        k = 0.5 * (1.0 + math.cos(math.pi * progress))
        if increasing:
            return (1-k) * self.lambda_multimodal_contrastive_loss_
        else:
            return k * self.lambda_multimodal_contrastive_loss_

    def train(self,
              n_epochs: int=100,
              n_epochs_all_gps: int=25,
              n_epochs_no_edge_recon: int=0,
              n_epochs_no_cat_covariates_contrastive: int=5,
              lr: float=0.001,
              weight_decay: float=0.,
              lambda_edge_recon: Optional[float]=500000.,
              lambda_cat_covariates_contrastive: Optional[float]=0.,
              lambda_multimodal_contrastive_loss: float=0.,
              multimodal_temperature: float=1.0,
              multimodal_contrastive_anneal: bool=False,
              contrastive_logits_pos_ratio: Optional[float]=0.125,
              contrastive_logits_neg_ratio: Optional[float]=0.125,
              lambda_gene_expr_recon: float=100.,
              lambda_chrom_access_recon: float=10.,
              lambda_group_lasso: float=0.,
              lambda_l1_masked: float=0.,
              l1_targets_mask: Optional[torch.Tensor]=None,
              l1_sources_mask: Optional[torch.Tensor]=None,
              lambda_l1_addon: float=0.,
              mlflow_experiment_id: Optional[str]=None):
        """
        Train the CustomNicheCompass model.
        """
        self.n_epochs_ = n_epochs
        self.n_epochs_all_gps_ = n_epochs_all_gps
        self.n_epochs_no_edge_recon_ = n_epochs_no_edge_recon
        self.n_epochs_no_cat_covariates_contrastive_ = (
            n_epochs_no_cat_covariates_contrastive)
        self.lr_ = lr
        self.weight_decay_ = weight_decay
        self.lambda_edge_recon_ = lambda_edge_recon
        self.lambda_gene_expr_recon_ = lambda_gene_expr_recon
        self.lambda_chrom_access_recon_ = lambda_chrom_access_recon
        self.lambda_cat_covariates_contrastive_ = (
            lambda_cat_covariates_contrastive)
        self.lambda_multimodal_contrastive_loss_ = (
            lambda_multimodal_contrastive_loss)
        self.multimodal_temperature_ = multimodal_temperature
        self.multimodal_contrastive_anneal_ = multimodal_contrastive_anneal
        self.contrastive_logits_pos_ratio_ = contrastive_logits_pos_ratio
        self.contrastive_logits_neg_ratio_ = contrastive_logits_neg_ratio
        self.lambda_group_lasso_ = lambda_group_lasso
        self.lambda_l1_masked_ = lambda_l1_masked
        self.l1_targets_mask = l1_targets_mask
        self.l1_sources_mask = l1_sources_mask
        self.lambda_l1_addon_ = lambda_l1_addon
        self.mlflow_experiment_id = mlflow_experiment_id

        print("\n--- MODEL TRAINING ---")

        if self.mlflow_experiment_id is not None:
            for attr, attr_value in self._get_public_attributes().items():
                mlflow.log_param(attr, attr_value)
            self.model.log_module_hyperparams_to_mlflow()

        start_time = time.time()
        self.epoch_logs = defaultdict(list)
        self.model.train()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params,
                                          lr=lr,
                                          weight_decay=weight_decay)

        for self.epoch in range(n_epochs):
            if self.epoch < self.n_epochs_no_edge_recon_:
                self.edge_recon_active = False
            else:
                self.edge_recon_active = True
            if self.epoch < self.n_epochs_all_gps_:
                self.use_only_active_gps = False
            else:
                self.use_only_active_gps = True
            if self.epoch < self.n_epochs_no_cat_covariates_contrastive_:
                self.cat_covariates_contrastive_active = False
            else:
                self.cat_covariates_contrastive_active = True

            self.multimodal_contrastive_weight_ = (
                self._get_multimodal_contrastive_weight())

            self.iter_logs = defaultdict(list)
            self.iter_logs["n_train_iter"] = 0
            self.iter_logs["n_val_iter"] = 0

            for edge_train_data_batch, node_train_data_batch in zip(
                    self.edge_train_loader,
                    _cycle_iterable(self.node_train_loader)):
                node_train_data_batch = node_train_data_batch.to(self.device)
                node_train_model_output = self.model(
                    data_batch=node_train_data_batch,
                    decoder="omics",
                    use_only_active_gps=self.use_only_active_gps)

                edge_train_data_batch = edge_train_data_batch.to(self.device)
                edge_train_model_output = self.model(
                    data_batch=edge_train_data_batch,
                    decoder="graph",
                    use_only_active_gps=self.use_only_active_gps)

                train_loss_dict = self.model.loss(
                    edge_model_output=edge_train_model_output,
                    node_model_output=node_train_model_output,
                    lambda_edge_recon=self.lambda_edge_recon_,
                    lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
                    lambda_chrom_access_recon=self.lambda_chrom_access_recon_,
                    lambda_cat_covariates_contrastive=self.lambda_cat_covariates_contrastive_,
                    lambda_multimodal_contrastive_loss=self.multimodal_contrastive_weight_,
                    multimodal_temperature=self.multimodal_temperature_,
                    multimodal_contrastive_active=(
                        self.multimodal_contrastive_weight_ > 0),
                    contrastive_logits_pos_ratio=self.contrastive_logits_pos_ratio_,
                    contrastive_logits_neg_ratio=self.contrastive_logits_neg_ratio_,
                    lambda_group_lasso=self.lambda_group_lasso_,
                    lambda_l1_masked=self.lambda_l1_masked_,
                    l1_targets_mask=self.l1_targets_mask,
                    l1_sources_mask=self.l1_sources_mask,
                    lambda_l1_addon=self.lambda_l1_addon_,
                    edge_recon_active=self.edge_recon_active,
                    cat_covariates_contrastive_active=self.cat_covariates_contrastive_active)

                train_global_loss = train_loss_dict["global_loss"]
                train_optim_loss = train_loss_dict["optim_loss"]

                if self.verbose_:
                    for key, value in train_loss_dict.items():
                        self.iter_logs[f"train_{key}"].append(value.item())
                else:
                    self.iter_logs["train_global_loss"].append(
                        train_global_loss.item())
                    self.iter_logs["train_optim_loss"].append(
                        train_optim_loss.item())
                    # Always log multimodal_contrastive_loss if present
                    if "multimodal_contrastive_loss" in train_loss_dict:
                        self.iter_logs["train_multimodal_contrastive_loss"].append(
                            train_loss_dict["multimodal_contrastive_loss"].item())
                self.iter_logs["n_train_iter"] += 1

                self.optimizer.zero_grad()
                train_optim_loss.backward()
                if self.grad_clip_value_ > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                    self.grad_clip_value_)
                self.optimizer.step()

            if (self.edge_val_loader is not None and
                self.node_val_loader is not None):
                self.eval_epoch()
            elif (self.edge_val_loader is None and
            self.node_val_loader is not None):
                warnings.warn("You have specified a node validation set but no "
                              "edge validation set. Skipping validation...")
            elif (self.edge_val_loader is not None and
            self.node_val_loader is None):
                warnings.warn("You have specified an edge validation set but no"
                              " node validation set. Skipping validation...")

            for key in self.iter_logs:
                if key.startswith("train"):
                    epoch_avg_loss = (
                        np.array(self.iter_logs[key]).sum() /
                        self.iter_logs["n_train_iter"])
                    self.epoch_logs[key].append(epoch_avg_loss)
                    # Log training losses to MLflow
                    if self.mlflow_experiment_id is not None:
                        mlflow.log_metric(key, epoch_avg_loss, step=self.epoch)
                if key.startswith("val"):
                    epoch_avg_loss = (
                        np.array(self.iter_logs[key]).sum() /
                        self.iter_logs["n_val_iter"])
                    self.epoch_logs[key].append(epoch_avg_loss)
                    # Log validation losses to MLflow
                    if self.mlflow_experiment_id is not None:
                        mlflow.log_metric(key, epoch_avg_loss, step=self.epoch)

            if self.monitor_:
                print_progress(self.epoch, self.epoch_logs, self.n_epochs_)

            if self.use_early_stopping_:
                if self.is_early_stopping():
                    break

        self.training_time += (time.time() - start_time)
        minutes, seconds = divmod(self.training_time, 60)
        print(f"Model training finished after {int(minutes)} min {int(seconds)}"
              " sec.")
        if self.best_model_state_dict is not None and self.reload_best_model_:
            print("Using best model state, which was in epoch "
                  f"{self.best_epoch + 1}.")
            self.model.load_state_dict(self.best_model_state_dict)

        self.model.eval()

        if self.edge_val_loader is not None:
            self.eval_end()

    @torch.no_grad()
    def eval_epoch(self):
        """
        Epoch evaluation logic of CustomNicheCompass model used during training.
        """
        self.model.eval()

        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])
        edge_same_cat_covariates_cat_val_accumulated = [
            np.array([]) for _ in range(self.n_cat_covariates)]
        edge_incl_val_accumulated = np.array([])

        multimodal_contrastive_weight = self._get_multimodal_contrastive_weight()

        for edge_val_data_batch, node_val_data_batch in zip(
                self.edge_val_loader, _cycle_iterable(self.node_val_loader)):
            node_val_data_batch = node_val_data_batch.to(self.device)
            node_val_model_output = self.model(
                data_batch=node_val_data_batch,
                decoder="omics",
                use_only_active_gps=self.use_only_active_gps)

            edge_val_data_batch = edge_val_data_batch.to(self.device)
            edge_val_model_output = self.model(
                data_batch=edge_val_data_batch,
                decoder="graph",
                use_only_active_gps=self.use_only_active_gps)

            val_loss_dict = self.model.loss(
                edge_model_output=edge_val_model_output,
                node_model_output=node_val_model_output,
                lambda_edge_recon=self.lambda_edge_recon_,
                lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
                lambda_chrom_access_recon=self.lambda_chrom_access_recon_,
                lambda_cat_covariates_contrastive=self.lambda_cat_covariates_contrastive_,
                lambda_multimodal_contrastive_loss=multimodal_contrastive_weight,
                multimodal_temperature=self.multimodal_temperature_,
                multimodal_contrastive_active=(multimodal_contrastive_weight > 0),
                contrastive_logits_pos_ratio=self.contrastive_logits_pos_ratio_,
                contrastive_logits_neg_ratio=self.contrastive_logits_neg_ratio_,
                lambda_group_lasso=self.lambda_group_lasso_,
                lambda_l1_masked=self.lambda_l1_masked_,
                l1_targets_mask=self.l1_targets_mask,
                l1_sources_mask=self.l1_sources_mask,
                lambda_l1_addon=self.lambda_l1_addon_,
                edge_recon_active=True)

            val_global_loss = val_loss_dict["global_loss"]
            val_optim_loss = val_loss_dict["optim_loss"]
            if self.verbose_:
                for key, value in val_loss_dict.items():
                    self.iter_logs[f"val_{key}"].append(value.item())
            else:
                self.iter_logs["val_global_loss"].append(val_global_loss.item())
                self.iter_logs["val_optim_loss"].append(val_optim_loss.item())
                # Always log multimodal_contrastive_loss if present
                if "multimodal_contrastive_loss" in val_loss_dict:
                    self.iter_logs["val_multimodal_contrastive_loss"].append(
                        val_loss_dict["multimodal_contrastive_loss"].item())
            self.iter_logs["n_val_iter"] += 1

            edge_recon_probs_val = torch.sigmoid(
                edge_val_model_output["edge_recon_logits"])
            edge_recon_labels_val = edge_val_model_output["edge_recon_labels"]
            edge_same_cat_covariates_cat_val = (
                edge_val_model_output["edge_same_cat_covariates_cat"])
            edge_incl_val = edge_val_model_output["edge_incl"]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())
            if edge_same_cat_covariates_cat_val is not None:
                for i, edge_same_cat_covariate_cat_val in enumerate(
                        edge_same_cat_covariates_cat_val):
                    edge_same_cat_covariates_cat_val_accumulated[i] = np.append(
                        edge_same_cat_covariates_cat_val_accumulated[i],
                        edge_same_cat_covariate_cat_val.detach().cpu().numpy())
            if edge_incl_val is not None:
                edge_incl_val_accumulated = np.append(
                    edge_incl_val_accumulated,
                    edge_incl_val.detach().cpu().numpy())
            else:
                edge_same_cat_covariates_cat_val_accumulated = None
                edge_incl_val_accumulated = None
        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated,
            edge_same_cat_covariates_cat=edge_same_cat_covariates_cat_val_accumulated,
            edge_incl=edge_incl_val_accumulated)
        if self.verbose_:
            self.epoch_logs["val_auroc_score"].append(
                val_eval_dict["auroc_score"])
            self.epoch_logs["val_auprc_score"].append(
                val_eval_dict["auprc_score"])
            self.epoch_logs["val_best_acc_score"].append(
                val_eval_dict["best_acc_score"])
            self.epoch_logs["val_best_f1_score"].append(
                val_eval_dict["best_f1_score"])

        # Log evaluation metrics to MLflow during training
        if self.mlflow_experiment_id is not None:
            for key, value in val_eval_dict.items():
                mlflow.log_metric(f"val_iters_{key}", value, step=self.epoch)

        self.model.train()

class CustomVGPGAE(VGPGAE):
    """
    Project-specific VGPGAE with custom behavior.

    Override methods as needed; this is a safe default that calls the parent
    implementation while giving you a single place to modify logic.
    """
    def __init__(self, *args, **kwargs):

        # Remove 'multimodal_embedding_size' from kwargs before passing to super().__init__
        kwargs_no_mme = dict(kwargs)
        kwargs_no_mme.pop("multimodal_embedding_size", None)
        super().__init__(*args, **kwargs_no_mme)

        self.multimodal_embedding_size_ = kwargs.get("multimodal_embedding_size")

        n_cat_covariates_embed_input = (
            sum(self.cat_covariates_embeds_nums_)
            if ("encoder" in self.cat_covariates_embeds_injection_) &
            (self.n_cat_covariates_ > 0)
            else 0
        )

        # Separate encoders for RNA and ATAC inputs when available.
        self.encoder_rna = Encoder(
            n_input=self.n_output_genes_,
            n_cat_covariates_embed_input=n_cat_covariates_embed_input,
            n_fc_layers=self.n_fc_layers_encoder_,
            n_layers=self.n_layers_encoder_,
            n_hidden=self.n_hidden_encoder_,
            n_latent=self.n_prior_gp_,
            n_addon_latent=self.n_addon_gp_,
            conv_layer=self.conv_layer_encoder_,
            n_attention_heads=self.encoder_n_attention_heads_,
            dropout_rate=self.dropout_rate_encoder_,
            activation=torch.relu,
            use_bn=self.encoder_use_bn_)

        if self.n_output_peaks_ > 0:
            self.encoder_atac = Encoder(
                n_input=self.n_output_peaks_,
                n_cat_covariates_embed_input=n_cat_covariates_embed_input,
                n_fc_layers=self.n_fc_layers_encoder_,
                n_layers=self.n_layers_encoder_,
                n_hidden=self.n_hidden_encoder_,
                n_latent=self.n_prior_gp_,
                n_addon_latent=self.n_addon_gp_,
                conv_layer=self.conv_layer_encoder_,
                n_attention_heads=self.encoder_n_attention_heads_,
                dropout_rate=self.dropout_rate_encoder_,
                activation=torch.relu,
                use_bn=self.encoder_use_bn_)
        else:
            self.encoder_atac = None

        # Multimodal layer
        gp_embedding_size = self.n_prior_gp_ + self.n_addon_gp_
        self.multimodal_layer = torch.nn.Linear(gp_embedding_size, self.multimodal_embedding_size_)


    def multiply_gaussians_log_space(self,
                                     mu_rna: torch.Tensor,
                                     logstd_rna: torch.Tensor,
                                     mu_atac: torch.Tensor,
                                     logstd_atac: torch.Tensor) -> tuple:
        """
        Combines two Gaussians using log-standard-deviations.
        s1, s2: natural log of the standard deviation
        """
        # Calculate variances in log-space for stability
        # var = exp(2*s)
        v1 = torch.exp(2 * logstd_rna)
        v2 = torch.exp(2 * logstd_atac)
        denom = v1 + v2
        
        # New Mean
        mu_new = (mu_rna * v2 + mu_atac * v1) / denom
        
        # New Log-Std: s_new = s1 + s2 - 0.5 * ln(exp(2*s1) + exp(2*s2))
        # We use np.logaddexp for numerical stability
        s_new = logstd_rna + logstd_atac - 0.5 * torch.logaddexp(
            2 * logstd_rna, 2 * logstd_atac)
        
        return mu_new, s_new

    def _combine_posteriors(self,
                            mu_rna: torch.Tensor,
                            logstd_rna: torch.Tensor,
                            mu_atac: torch.Tensor,
                            logstd_atac: torch.Tensor,
                            modality_mask: Optional[torch.Tensor]=None
                            ) -> tuple:
        if modality_mask is None:
            return self.multiply_gaussians_log_space(
                mu_rna, logstd_rna, mu_atac, logstd_atac)

        if modality_mask.dtype != torch.bool:
            modality_mask = modality_mask.to(torch.bool)
        has_rna = modality_mask[:, 0]
        has_atac = modality_mask[:, 1]
        both = has_rna & has_atac
        only_rna = has_rna & ~has_atac
        only_atac = ~has_rna & has_atac

        mu = torch.zeros_like(mu_rna)
        logstd = torch.zeros_like(logstd_rna)

        if both.any():
            mu_both, logstd_both = self.multiply_gaussians_log_space(
                mu_rna[both],
                logstd_rna[both],
                mu_atac[both],
                logstd_atac[both])
            mu[both] = mu_both
            logstd[both] = logstd_both
        if only_rna.any():
            mu[only_rna] = mu_rna[only_rna]
            logstd[only_rna] = logstd_rna[only_rna]
        if only_atac.any():
            mu[only_atac] = mu_atac[only_atac]
            logstd[only_atac] = logstd_atac[only_atac]

        return mu, logstd

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

        if self.encoder_atac is None:
            return super().forward(
                data_batch=data_batch,
                decoder=decoder,
                use_only_active_gps=use_only_active_gps,
                return_agg_weights=return_agg_weights,
                update_atac_dynamic_decoder_mask=update_atac_dynamic_decoder_mask,
            )

        output = {}

        # Separate rna and atac data
        x_enc_rna = x_enc[:, :self.n_output_genes_]
        x_enc_atac = x_enc[:, self.n_output_genes_:]
        
        # Use encoder to get latent distribution parameters for current batch
        # and reparameterization trick to get latent features (gp scores).
        # Filter for nodes in current batch

        # Encode rna data
        encoder_outputs_rna = self.encoder_rna(
            x=x_enc_rna,
            edge_index=edge_index,
            cat_covariates_embed=(self.cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        self.mu_rna = encoder_outputs_rna[0][batch_idx, :]
        self.logstd_rna = encoder_outputs_rna[1][batch_idx, :]
        output["mu_rna"] = self.mu_rna
        output["logstd_rna"] = self.logstd_rna
        z_rna = self.reparameterize(self.mu_rna, self.logstd_rna)

        # Encode atac data
        encoder_outputs_atac = self.encoder_atac(
            x=x_enc_atac,
            edge_index=edge_index,
            cat_covariates_embed=(self.cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        self.mu_atac = encoder_outputs_atac[0][batch_idx, :]
        self.logstd_atac = encoder_outputs_atac[1][batch_idx, :]
        output["mu_atac"] = self.mu_atac
        output["logstd_atac"] = self.logstd_atac
        z_atac = self.reparameterize(self.mu_atac, self.logstd_atac)

        modality_mask = getattr(data_batch, "modality_mask", None)
        if modality_mask is not None:
            modality_mask = modality_mask[batch_idx]

        # Combine rna and atac latent distributions using product of Gaussians
        self.mu, self.logstd = self._combine_posteriors(
            self.mu_rna,
            self.logstd_rna,
            self.mu_atac,
            self.logstd_atac,
            modality_mask=modality_mask)
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


    def get_latent_representation(
            self,
            node_batch: Data,
            only_active_gps: bool=True,
            return_mu_std: bool=False
            ) -> torch.Tensor:
        """
        Encode RNA + ATAC separately and combine latents via Gaussian product.
        """
        if self.encoder_atac is None:
            return super().get_latent_representation(
                node_batch=node_batch,
                only_active_gps=only_active_gps,
                return_mu_std=return_mu_std)

        if self.log_variational_:
            x_enc = torch.log(1 + node_batch.x)
        else:
            x_enc = node_batch.x

        x_enc_rna = x_enc[:, :self.n_output_genes_]
        x_enc_atac = x_enc[:, self.n_output_genes_:]

        if len(self.cat_covariates_cats_) > 0:
            cat_covariates_embeds = []
            for i in range(len(self.cat_covariates_embedders)):
                cat_covariates_embeds.append(self.cat_covariates_embedders[i](
                    node_batch.cat_covariates_cats[:, i]))
                cat_covariates_embed = torch.cat(
                    cat_covariates_embeds,
                    dim=1)
        else:
            cat_covariates_embed = None

        encoder_outputs_rna = self.encoder_rna(
            x=x_enc_rna,
            edge_index=node_batch.edge_index,
            cat_covariates_embed=(cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        mu_rna = encoder_outputs_rna[0][:node_batch.batch_size, :]
        logstd_rna = encoder_outputs_rna[1][:node_batch.batch_size, :]

        encoder_outputs_atac = self.encoder_atac(
            x=x_enc_atac,
            edge_index=node_batch.edge_index,
            cat_covariates_embed=(cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        mu_atac = encoder_outputs_atac[0][:node_batch.batch_size, :]
        logstd_atac = encoder_outputs_atac[1][:node_batch.batch_size, :]

        modality_mask = getattr(node_batch, "modality_mask", None)
        if modality_mask is not None:
            modality_mask = modality_mask[:node_batch.batch_size]
        mu, logstd = self._combine_posteriors(
            mu_rna,
            logstd_rna,
            mu_atac,
            logstd_atac,
            modality_mask=modality_mask)

        if only_active_gps:
            active_gp_mask = self.get_active_gp_mask()
            mu, logstd = mu[:, active_gp_mask], logstd[:, active_gp_mask]

        if return_mu_std:
            std = torch.exp(logstd)
            return mu, std
        z = self.reparameterize(mu, logstd)
        return z


    def get_omics_decoder_outputs(
            self,
            node_batch: Data,
            only_active_gps: bool=True,
            ) -> dict:
        """
        Decode using a combined RNA+ATAC latent (product-of-Gaussians).

        Mirrors `nichecompass.modules.vgpgae.VGPGAE.get_omics_decoder_outputs`,
        but uses `encoder_rna` / `encoder_atac` and combines their posteriors
        before decoding.
        """
        if self.encoder_atac is None:
            return super().get_omics_decoder_outputs(
                node_batch=node_batch,
                only_active_gps=only_active_gps)

        x = node_batch.x  # dim: n_obs x n_omics_features
        edge_index = node_batch.edge_index
        batch_idx = slice(None, node_batch.batch_size)

        # Logarithmitize omics feature vector if done during training
        if self.log_variational_:
            x_enc = torch.log(1 + x)
        else:
            x_enc = x

        # Get categorical covariate embeddings
        if len(self.cat_covariates_cats_) > 0:
            cat_covariates_embeds = []
            for i in range(len(self.cat_covariates_embedders)):
                cat_covariates_embeds.append(self.cat_covariates_embedders[i](
                    node_batch.cat_covariates_cats[:, i]))
                cat_covariates_embed = torch.cat(
                    cat_covariates_embeds,
                    dim=1)
        else:
            cat_covariates_embed = None

        # Split encoder inputs by modality
        x_enc_rna = x_enc[:, :self.n_output_genes_]
        x_enc_atac = x_enc[:, self.n_output_genes_:]

        # Encode RNA + ATAC and combine posteriors
        encoder_outputs_rna = self.encoder_rna(
            x=x_enc_rna,
            edge_index=edge_index,
            cat_covariates_embed=(cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        mu_rna = encoder_outputs_rna[0][batch_idx, :]
        logstd_rna = encoder_outputs_rna[1][batch_idx, :]

        encoder_outputs_atac = self.encoder_atac(
            x=x_enc_atac,
            edge_index=edge_index,
            cat_covariates_embed=(cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        mu_atac = encoder_outputs_atac[0][batch_idx, :]
        logstd_atac = encoder_outputs_atac[1][batch_idx, :]

        modality_mask = getattr(node_batch, "modality_mask", None)
        if modality_mask is not None:
            modality_mask = modality_mask[batch_idx]
        mu, logstd = self._combine_posteriors(
            mu_rna,
            logstd_rna,
            mu_atac,
            logstd_atac,
            modality_mask=modality_mask)
        z = self.reparameterize(mu, logstd)

        if only_active_gps:
            active_gp_mask = self.get_active_gp_mask()
            # Set gp scores of inactive gene programs to 0 to not affect decoders
            z[:, ~active_gp_mask] = 0

        output = {}
        output["node_labels"] = {}

        # Get rna and atac part from omics feature vector
        x_atac = x[:, self.n_output_genes_:]
        x = x[:, :self.n_output_genes_]

        # Compute aggregated neighborhood rna feature vector
        rna_node_label_aggregator_output = self.rna_node_label_aggregator(
                x=x,
                edge_index=edge_index,
                return_agg_weights=False)
        x_neighbors = rna_node_label_aggregator_output[0]

        # Retrieve rna node labels and only keep nodes in current node batch
        assert x.size(1) == self.n_output_genes_
        assert x_neighbors.size(1) == self.n_output_genes_
        output["node_labels"]["target_rna"] = x[batch_idx]
        output["node_labels"]["source_rna"] = x_neighbors[batch_idx]

        # Use observed library size as scaling factor for the NB means
        target_rna_library_size = output["node_labels"]["target_rna"].sum(
            1).unsqueeze(1)
        source_rna_library_size = output["node_labels"]["source_rna"].sum(
            1).unsqueeze(1)
        target_rna_log_library_size = torch.log(target_rna_library_size)
        source_rna_log_library_size = torch.log(source_rna_library_size)

        output["target_rna_nb_means"] = self.target_rna_decoder(
            z=z,
            log_library_size=target_rna_log_library_size,
            cat_covariates_embed=(
                cat_covariates_embed[batch_idx] if
                (cat_covariates_embed is not None) &
                ("gene_expr_decoder" in
                 self.cat_covariates_embeds_injection_)
                 else None))
        output["source_rna_nb_means"] = self.source_rna_decoder(
            z=z,
            log_library_size=source_rna_log_library_size,
            cat_covariates_embed=(
            cat_covariates_embed[batch_idx] if
            (cat_covariates_embed is not None) &
            ("gene_expr_decoder" in
             self.cat_covariates_embeds_injection_)
             else None))

        if "atac" in self.modalities_:
            # Compute aggregated neighborhood atac feature vector
            atac_node_label_aggregator_output = (
                self.atac_node_label_aggregator(
                    x=x_atac,
                    edge_index=edge_index,
                    return_agg_weights=False))
            x_neighbors_atac = atac_node_label_aggregator_output[0]

            # Retrieve node labels and only keep nodes in current node batch
            assert x_atac.size(1) == self.n_output_peaks_
            assert x_neighbors_atac.size(1) == self.n_output_peaks_
            output["node_labels"]["target_atac"] = x_atac[batch_idx][
                :, self.features_idx_dict_["target_reconstructed_atac_idx"]]
            output["node_labels"]["source_atac"] = x_neighbors_atac[batch_idx][
                :, self.features_idx_dict_["source_reconstructed_atac_idx"]]

            # Use observed library size as scaling factor for the NB means
            target_atac_library_size = output["node_labels"][
                "target_atac"].sum(1).unsqueeze(1)
            source_atac_library_size = output["node_labels"][
                "source_atac"].sum(1).unsqueeze(1)
            target_atac_log_library_size = torch.log(target_atac_library_size)
            source_atac_log_library_size = torch.log(source_atac_library_size)

            output["target_atac_nb_means"] = self.target_atac_decoder(
                z=z,
                log_library_size=target_atac_log_library_size,
                dynamic_mask=self.target_atac_dynamic_decoder_mask,
                cat_covariates_embed=(
                    cat_covariates_embed[batch_idx] if
                    (cat_covariates_embed is not None) &
                    ("chrom_access_decoder" in
                     self.cat_covariates_embeds_injection_) else
                    None))[
                :, self.features_idx_dict_["target_reconstructed_atac_idx"]]
            output["source_atac_nb_means"] = self.source_atac_decoder(
                z=z,
                log_library_size=source_atac_log_library_size,
                dynamic_mask=self.source_atac_dynamic_decoder_mask,
                cat_covariates_embed=(
                    cat_covariates_embed[batch_idx] if
                    (cat_covariates_embed is not None) &
                    ("chrom_access_decoder" in
                     self.cat_covariates_embeds_injection_) else
                    None))[
                :, self.features_idx_dict_["source_reconstructed_atac_idx"]]
        return output

    def loss(self,
             edge_model_output: dict,
             node_model_output: dict,
             lambda_l1_masked: float,
             l1_targets_mask: torch.Tensor,
             l1_sources_mask: torch.Tensor,
             lambda_l1_addon: float,
             lambda_group_lasso: float,
             lambda_gene_expr_recon: float=300.,
             lambda_chrom_access_recon: float=100.,
             lambda_edge_recon: Optional[float]=500000.,
             lambda_cat_covariates_contrastive: Optional[float]=100000.,
             contrastive_logits_pos_ratio: float=0.125,
             contrastive_logits_neg_ratio: float=0.,
             edge_recon_active: bool=True,
             cat_covariates_contrastive_active: bool=True,
             lambda_multimodal_contrastive_loss: float=0.,
             multimodal_temperature: float=1.0,
             multimodal_contrastive_active: bool=True) -> dict:
        loss_dict = super().loss(
            edge_model_output=edge_model_output,
            node_model_output=node_model_output,
            lambda_l1_masked=lambda_l1_masked,
            l1_targets_mask=l1_targets_mask,
            l1_sources_mask=l1_sources_mask,
            lambda_l1_addon=lambda_l1_addon,
            lambda_group_lasso=lambda_group_lasso,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_chrom_access_recon=lambda_chrom_access_recon,
            lambda_edge_recon=lambda_edge_recon,
            lambda_cat_covariates_contrastive=lambda_cat_covariates_contrastive,
            contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
            contrastive_logits_neg_ratio=contrastive_logits_neg_ratio,
            edge_recon_active=edge_recon_active,
            cat_covariates_contrastive_active=cat_covariates_contrastive_active)

        if ("atac" in self.modalities_ and
                lambda_multimodal_contrastive_loss > 0 and
                multimodal_contrastive_active and
                "mu_rna" in node_model_output and
                "mu_atac" in node_model_output):
            similarity_matrix = self.get_multimodal_similarity(
                mu_rna=node_model_output["mu_rna"],
                mu_atac=node_model_output["mu_atac"])
            loss_dict["multimodal_contrastive_loss"] = (
                lambda_multimodal_contrastive_loss *
                self.compute_multimodal_contrastive_loss(
                    similarity_matrix,
                    temperature=multimodal_temperature))
            loss_dict["global_loss"] += loss_dict[
                "multimodal_contrastive_loss"]
            loss_dict["optim_loss"] += loss_dict[
                "multimodal_contrastive_loss"]

        return loss_dict

    def get_multimodal_similarity(self,
                                  mu_rna: Optional[torch.Tensor]=None,
                                  mu_atac: Optional[torch.Tensor]=None
                                  ) -> torch.Tensor:
        mu_rna = self.mu_rna if mu_rna is None else mu_rna
        mu_atac = self.mu_atac if mu_atac is None else mu_atac

        mu_rna = self.multimodal_layer(mu_rna)
        mu_atac = self.multimodal_layer(mu_atac)

        mu_rna_normed = F.normalize(mu_rna, p=2, dim=1)
        mu_atac_normed = F.normalize(mu_atac, p=2, dim=1)

        similarity_matrix = torch.matmul(mu_rna_normed, mu_atac_normed.t())
        return similarity_matrix

    def compute_multimodal_contrastive_loss(
            self,
            similarity_matrix: torch.Tensor,
            temperature: float=1.0) -> torch.Tensor:
        temperature = max(temperature, 1e-8)
        logits = similarity_matrix / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_rna = F.cross_entropy(logits, labels)
        loss_atac = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_rna + loss_atac)

    def add_multimodal_contrastive_loss(
            self,
            mu_rna: Optional[torch.Tensor]=None,
            mu_atac: Optional[torch.Tensor]=None,
            temperature: float=1.0) -> torch.Tensor:
        similarity_matrix = self.get_multimodal_similarity(
            mu_rna=mu_rna,
            mu_atac=mu_atac)
        return self.compute_multimodal_contrastive_loss(
            similarity_matrix,
            temperature=temperature)


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
                 cat_covariates_keys: Optional[List[str]]=None,
                 paired_data: bool=True):
        if counts_key is None:
            x_rna = adata.X
        else:
            x_rna = adata.layers[counts_key]

        # Store features in dense format
        if sp.issparse(x_rna): 
            self.x_rna = torch.tensor(x_rna.toarray())
        else:
            self.x_rna = torch.tensor(x_rna)

        # Store ATAC features in dense format if provided
        if adata_atac is not None:
            if paired_data:
                if ((adata.n_obs != adata_atac.n_obs) or
                    (not adata.obs_names.equals(adata_atac.obs_names))):
                    raise ValueError(
                        "adata and adata_atac must have matching obs_names in "
                        "the same order when paired_data=True. Set "
                        "paired_data=False for unpaired forward passes.")

                if sp.issparse(adata_atac.X):
                    self.x_atac = torch.tensor(adata_atac.X.toarray())
                else:
                    self.x_atac = torch.tensor(adata_atac.X)
                self.x = torch.cat((self.x_rna, self.x_atac), axis=1)
                self.modality_mask = torch.ones(
                    (self.x.size(0), 2), dtype=torch.bool)
            else:
                rna_obs = adata.obs_names
                atac_obs = adata_atac.obs_names
                extra_atac_obs = atac_obs[~atac_obs.isin(rna_obs)]
                union_obs = rna_obs.append(extra_atac_obs)
                n_obs_total = len(union_obs)

                rna_positions = np.arange(adata.n_obs)
                atac_positions = union_obs.get_indexer(atac_obs)
                if (atac_positions < 0).any():
                    raise ValueError("Unpaired union index is missing ATAC obs.")

                x_rna_full = torch.zeros(
                    (n_obs_total, adata.n_vars),
                    dtype=self.x_rna.dtype)
                x_rna_full[rna_positions] = self.x_rna

                if sp.issparse(adata_atac.X):
                    x_atac = torch.tensor(adata_atac.X.toarray(),
                                          dtype=self.x_rna.dtype)
                else:
                    x_atac = torch.tensor(adata_atac.X,
                                          dtype=self.x_rna.dtype)
                x_atac_full = torch.zeros(
                    (n_obs_total, adata_atac.n_vars),
                    dtype=self.x_rna.dtype)
                x_atac_full[atac_positions] = x_atac

                self.x_rna = x_rna_full
                self.x_atac = x_atac_full
                self.x = torch.cat((self.x_rna, self.x_atac), axis=1)

                modality_mask = torch.zeros((n_obs_total, 2), dtype=torch.bool)
                modality_mask[rna_positions, 0] = True
                modality_mask[atac_positions, 1] = True
                self.modality_mask = modality_mask
        else:
            self.x_atac = None
            self.x = self.x_rna
            self.modality_mask = torch.stack(
                (torch.ones(self.x.size(0), dtype=torch.bool),
                 torch.zeros(self.x.size(0), dtype=torch.bool)),
                dim=1)

        # Store adjacency matrix in torch_sparse SparseTensor format
        if paired_data or adata_atac is None:
            adj_rna = (adata.obsp[adj_key] if sp.issparse(adata.obsp[adj_key])
                       else sp.csr_matrix(adata.obsp[adj_key]))
            self.adj = sparse_mx_to_sparse_tensor(adj_rna)
        else:
            adj_rna = (adata.obsp[adj_key] if sp.issparse(adata.obsp[adj_key])
                       else sp.csr_matrix(adata.obsp[adj_key]))
            if adj_key not in adata_atac.obsp:
                raise ValueError("Please specify an adequate 'adj_key' for "
                                 "adata_atac when paired_data=False.")
            atac_obs = adata_atac.obs_names
            atac_only_mask = ~atac_obs.isin(adata.obs_names)
            if atac_only_mask.any():
                atac_only_idx = np.where(atac_only_mask)[0]
                adj_atac_full = (adata_atac.obsp[adj_key] if sp.issparse(
                    adata_atac.obsp[adj_key]) else sp.csr_matrix(
                        adata_atac.obsp[adj_key]))
                adj_atac = adj_atac_full[atac_only_idx][:, atac_only_idx]
                adj = sp.block_diag((adj_rna, adj_atac), format="csr")
            else:
                adj = adj_rna
            self.adj = sparse_mx_to_sparse_tensor(adj)
            
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
                if paired_data or adata_atac is None:
                    cat_covariate_cats = torch.tensor(
                        encode_labels(adata,
                                      cat_covariate_label_encoder,
                                      cat_covariate_key),
                        dtype=torch.long)
                else:
                    cat_covariate_cats = -1 * np.ones(
                        self.x.size(0), dtype=np.int64)
                    rna_encoded = encode_labels(
                        adata, cat_covariate_label_encoder, cat_covariate_key)
                    rna_encoded = rna_encoded.astype(np.int64)
                    cat_covariate_cats[:adata.n_obs] = rna_encoded
                    atac_obs = adata_atac.obs_names
                    atac_only_mask = ~atac_obs.isin(adata.obs_names)
                    if atac_only_mask.any():
                        atac_encoded = encode_labels(
                            adata_atac,
                            cat_covariate_label_encoder,
                            cat_covariate_key).astype(np.int64)
                        extra_positions = np.arange(
                            adata.n_obs, self.x.size(0))
                        cat_covariate_cats[extra_positions] = (
                            atac_encoded[atac_only_mask])
                    cat_covariate_cats = torch.tensor(
                        cat_covariate_cats, dtype=torch.long)
                self.cat_covariates_cats.append(cat_covariate_cats)
            self.cat_covariates_cats = torch.stack(self.cat_covariates_cats,
                                                   dim=1)            

        self.n_node_features = self.x.size(1)
        self.size_factors = self.x.sum(1) # fix for ATAC case

    def __len__(self):
        """Return the number of observations stored in SpatialAnnTorchDataset"""
        return self.x.size(0)


def prepare_data(adata: AnnData,
                 cat_covariates_label_encoders: List[dict],
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 cat_covariates_keys: Optional[List[str]]=None,
                 paired_data: bool=True,
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
        cat_covariates_label_encoders=cat_covariates_label_encoders,
        paired_data=paired_data)

    # PyG Data object (has 2 edge index pairs for one edge because of symmetry;
    # one edge index pair will be removed in the edge-level split).
    data = Data(
        x=dataset.x,
        edge_index=dataset.edge_index,
        edge_attr=dataset.edge_index.t()) # store index of edge nodes as
                                          # edge attribute for
                                          # aggregation weight retrieval
                                          # in mini batches

    # Keep per-modality views for debugging / custom logic.
    data.x_rna = dataset.x_rna
    if dataset.x_atac is not None:
        data.x_atac = dataset.x_atac

    data.modality_mask = dataset.modality_mask

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


__all__ = [
    "CustomNicheCompass",
    "CustomTrainer",
    "CustomVGPGAE",
    "CustomSpatialAnnTorchDataset",
    "prepare_data",
]
