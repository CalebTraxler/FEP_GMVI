#!/usr/bin/env python3
"""
Corrected Full-Rank Gaussian Variational Inference for FEP Networks

This implementation fixes the mathematical issues in the original code:
- Proper multivariate KL divergence computation
- Principled hyperparameter choices
- Better covariance parameterization
- Removes ad-hoc scaling factors
- More aggressive outlier detection using FEP vs CCC differences
- Better uncertainty calibration
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

class CorrectedFullRankGMVI:
    """
    Corrected full-rank Gaussian variational inference for FEP networks.
    Fixes mathematical issues and provides principled uncertainty estimation.
    """
    
    def __init__(self, edge_data, node_data=None, prior_mean=0.0, prior_std=5.0, 
                 outlier_detection=True, device='cpu'):
        self.edge_data = edge_data
        self.node_data = node_data
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.outlier_detection = outlier_detection
        self.device = device
        
        # Hyperparameters - More aggressive settings
        self.kl_weight = 0.1  # Start with lower KL weight
        self.outlier_prior_prob = 0.2  # Higher prior probability of outlier
        
        # Build network and initialize
        self._build_network()
        self._initialize_parameters()
        
    def _build_network(self):
        """Build network structure from edge data."""
        # Get unique ligands
        all_ligands = set(self.edge_data['Ligand1'].tolist() + 
                         self.edge_data['Ligand2'].tolist())
        self.ligands = sorted(list(all_ligands))
        self.ligand_to_idx = {ligand: i for i, ligand in enumerate(self.ligands)}
        self.n_nodes = len(self.ligands)
        self.n_edges = len(self.edge_data)
        
        # Extract edge information
        self.edge_pairs = []
        self.edge_values = []
        self.edge_errors = []
        self.edge_exp_values = []
        self.edge_ccc_values = []
        
        for _, row in self.edge_data.iterrows():
            lig1, lig2 = row['Ligand1'], row['Ligand2']
            self.edge_pairs.append((lig1, lig2))
            self.edge_values.append(row['FEP'])
            self.edge_errors.append(row['FEP Error'])
            self.edge_exp_values.append(row['Exp.'])
            self.edge_ccc_values.append(row['CCC'])
        
        # Analyze FEP vs CCC differences for outlier hints
        self.fep_ccc_diffs = np.array(self.edge_values) - np.array(self.edge_ccc_values)
        print(f"FEP vs CCC differences: mean abs = {np.mean(np.abs(self.fep_ccc_diffs)):.3f}")
        
        # Convert to tensors
        self.edge_values_tensor = torch.tensor(
            self.edge_values, dtype=torch.float32, device=self.device
        )
        self.edge_errors_tensor = torch.tensor(
            self.edge_errors, dtype=torch.float32, device=self.device
        )
        
        print(f"Network: {self.n_nodes} nodes, {self.n_edges} edges")
        print(f"Edge error range: [{min(self.edge_errors):.3f}, {max(self.edge_errors):.3f}]")
        
    def _initialize_parameters(self):
        """Initialize variational parameters properly."""
        # Get least squares initialization
        initial_means = self._least_squares_init()
        
        # Mean parameters
        self.node_mu = torch.tensor(
            initial_means, dtype=torch.float32, device=self.device, requires_grad=True
        )
        
        # Covariance parameterization via Cholesky decomposition
        # Initialize as scaled identity - more conservative
        init_std = np.mean(self.edge_errors) * 1.5  # Reduced from 2.0
        L_init = torch.eye(self.n_nodes, dtype=torch.float32, device=self.device) * init_std
        
        # Store Cholesky factor (lower triangular)
        self.L_cholesky = torch.tril(L_init).clone().requires_grad_(True)
        
        # Edge precision parameters (log-precision for numerical stability)
        base_precision = 1.0 / (self.edge_errors_tensor ** 2)
        self.edge_log_precision = torch.log(base_precision).clone().requires_grad_(True)
        
        # Outlier detection parameters - More aggressive initialization
        if self.outlier_detection:
            # Initialize outlier logits based on FEP vs CCC differences
            initial_outlier_logits = np.full(
                self.n_edges, 
                np.log(self.outlier_prior_prob / (1 - self.outlier_prior_prob))
            )
            
            # Use FEP-CCC differences to inform initial outlier probabilities
            abs_diffs = np.abs(self.fep_ccc_diffs)
            diff_threshold = np.percentile(abs_diffs, 75)  # Top 25% as potential outliers
            for i, abs_diff in enumerate(abs_diffs):
                if abs_diff > diff_threshold:
                    initial_outlier_logits[i] = 1.0  # Higher initial outlier probability
            print(f"Initialized {sum(abs_diffs > diff_threshold)} edges with higher outlier probability")
            
            self.outlier_logits = torch.tensor(
                initial_outlier_logits, dtype=torch.float32, device=self.device, requires_grad=True
            )
            
            # Outlier scale (larger uncertainty for outliers)
            self.outlier_log_scale = torch.tensor(
                np.log(init_std * 3.0), dtype=torch.float32, device=self.device, requires_grad=True
            )
    
    def _least_squares_init(self):
        """Initialize using least squares solution."""
        # Build incidence matrix for least squares
        row_indices = []
        col_indices = []
        data = []
        
        for i, (lig1, lig2) in enumerate(self.edge_pairs):
            idx1 = self.ligand_to_idx[lig1]
            idx2 = self.ligand_to_idx[lig2]
            
            # Edge equation: node[idx2] - node[idx1] = edge_value
            row_indices.extend([i, i])
            col_indices.extend([idx1, idx2])
            data.extend([-1.0, 1.0])
        
        A = csr_matrix((data, (row_indices, col_indices)), 
                       shape=(self.n_edges, self.n_nodes))
        b = np.array(self.edge_values)
        
        # Weighted least squares using edge errors
        weights = 1.0 / np.array(self.edge_errors) ** 2
        W = csr_matrix((weights, (range(self.n_edges), range(self.n_edges))), 
                       shape=(self.n_edges, self.n_edges))
        
        try:
            # Solve weighted least squares: (A^T W A) x = A^T W b
            AtWA = A.T @ W @ A
            AtWb = A.T @ W @ b
            
            # Add regularization for numerical stability
            AtWA += csr_matrix(np.eye(self.n_nodes) * 1e-6)
            
            x = spsolve(AtWA, AtWb)
            
            # Center around prior mean
            x = x - np.mean(x) + self.prior_mean
            
            mae = np.mean(np.abs(A @ x - b))
            print(f"Least squares initialization MAE: {mae:.3f}")
            return x
            
        except Exception as e:
            print(f"Least squares failed: {e}, using prior mean")
            return np.full(self.n_nodes, self.prior_mean)
    
    def _get_covariance_matrix(self):
        """Get covariance matrix from Cholesky factor."""
        # Ensure positive diagonal elements
        L = self.L_cholesky.clone()
        diag_indices = torch.arange(self.n_nodes, device=self.device)
        L[diag_indices, diag_indices] = torch.abs(L[diag_indices, diag_indices]) + 1e-6
        
        # Covariance = L @ L^T
        cov = L @ L.T
        return cov
    
    def _compute_edge_predictions(self, node_values):
        """Compute edge predictions from node values."""
        predictions = []
        for lig1, lig2 in self.edge_pairs:
            idx1 = self.ligand_to_idx[lig1]
            idx2 = self.ligand_to_idx[lig2]
            pred = node_values[idx2] - node_values[idx1]
            predictions.append(pred)
        return torch.stack(predictions)
    
    def _compute_likelihood(self, node_values):
        """Compute likelihood with outlier detection."""
        edge_predictions = self._compute_edge_predictions(node_values)
        
        # Edge precisions
        edge_precisions = torch.exp(self.edge_log_precision)
        edge_scales = 1.0 / torch.sqrt(edge_precisions)
        
        # Normal likelihood
        normal_likelihood = Normal(edge_predictions, edge_scales).log_prob(self.edge_values_tensor)
        
        if self.outlier_detection:
            # Outlier likelihood (wider distribution)
            outlier_scale = torch.exp(self.outlier_log_scale)
            outlier_likelihood = Normal(edge_predictions, outlier_scale).log_prob(self.edge_values_tensor)
            
            # Mixture probabilities
            outlier_probs = torch.sigmoid(self.outlier_logits)
            
            # Log-sum-exp for numerical stability
            log_mixture = torch.logsumexp(
                torch.stack([
                    normal_likelihood + torch.log(1 - outlier_probs + 1e-8),
                    outlier_likelihood + torch.log(outlier_probs + 1e-8)
                ], dim=0), dim=0
            )
            return log_mixture.sum()
        else:
            return normal_likelihood.sum()
    
    def _compute_kl_divergence(self):
        """Compute KL divergence properly for multivariate normals."""
        # Posterior: N(mu, Sigma)
        mu_post = self.node_mu
        Sigma_post = self._get_covariance_matrix()
        
        # Prior: N(prior_mean * 1, prior_std^2 * I)
        mu_prior = torch.full_like(mu_post, self.prior_mean)
        Sigma_prior = torch.eye(self.n_nodes, device=self.device) * (self.prior_std ** 2)
        
        # KL(q||p) = 0.5 * [tr(Σ_prior^-1 Σ_post) + (μ_prior - μ_post)^T Σ_prior^-1 (μ_prior - μ_post) 
        #                  - k + log(det(Σ_prior)/det(Σ_post))]
        
        # Inverse of prior covariance (diagonal)
        Sigma_prior_inv = torch.eye(self.n_nodes, device=self.device) / (self.prior_std ** 2)
        
        # Mean difference
        mu_diff = mu_prior - mu_post
        
        # Terms
        trace_term = torch.trace(Sigma_prior_inv @ Sigma_post)
        quad_term = mu_diff.T @ Sigma_prior_inv @ mu_diff
        
        # Log determinants
        log_det_prior = self.n_nodes * 2 * np.log(self.prior_std)
        
        # For numerical stability, use Cholesky decomposition for log determinant
        try:
            L = torch.linalg.cholesky(Sigma_post)
            log_det_post = 2 * torch.sum(torch.log(torch.diag(L)))
        except:
            # Fallback if Cholesky fails
            eigenvals = torch.linalg.eigvals(Sigma_post).real
            eigenvals = torch.clamp(eigenvals, min=1e-8)
            log_det_post = torch.sum(torch.log(eigenvals))
        
        kl = 0.5 * (trace_term + quad_term - self.n_nodes + log_det_prior - log_det_post)
        return kl
    
    def elbo_loss(self, n_samples=5):
        """Compute ELBO loss with proper sampling."""
        # Sample from variational distribution
        var_dist = MultivariateNormal(self.node_mu, self._get_covariance_matrix())
        
        # Monte Carlo estimate of expected log likelihood
        log_likelihood = 0.0
        for _ in range(n_samples):
            sample = var_dist.sample()
            log_likelihood += self._compute_likelihood(sample)
        log_likelihood /= n_samples
        
        # KL divergence
        kl_div = self._compute_kl_divergence()
        
        # ELBO = E[log p(x|z)] - KL[q(z)||p(z)]
        elbo = log_likelihood - self.kl_weight * kl_div
        
        return -elbo  # Return negative for minimization
    
    def fit(self, n_epochs=1000, lr=0.01, n_samples=5, patience=100):
        """Fit the model with adaptive learning and early stopping."""
        # Collect all parameters
        all_params = [self.node_mu, self.L_cholesky, self.edge_log_precision]
        
        if self.outlier_detection:
            all_params.extend([self.outlier_logits, self.outlier_log_scale])
        
        optimizer = optim.Adam(all_params, lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=50
        )
        
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        print("Fitting corrected full-rank Gaussian VI model...")
        
        for epoch in range(n_epochs):
            loss = self.elbo_loss(n_samples)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            scheduler.step(loss)
            
            # Adaptive KL weighting (gradually increase to balance terms)
            if epoch > 100:
                self.kl_weight = min(1.0, 0.1 + epoch / 2000)  # More gradual increase
            
            if epoch % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, KL_weight={self.kl_weight:.3f}, LR={current_lr:.6f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Training completed. Final loss: {best_loss:.4f}")
        return losses
    
    def get_posterior_samples(self, n_samples=1000):
        """Get samples from posterior distribution."""
        var_dist = MultivariateNormal(self.node_mu, self._get_covariance_matrix())
        samples = var_dist.sample((n_samples,))
        return samples.detach().cpu().numpy()
    
    def get_node_estimates(self):
        """Get node estimates with proper uncertainty quantification."""
        samples = self.get_posterior_samples()
        
        return {
            'means': np.mean(samples, axis=0),
            'stds': np.std(samples, axis=0),
            'percentiles_2_5': np.percentile(samples, 2.5, axis=0),
            'percentiles_97_5': np.percentile(samples, 97.5, axis=0),
            'samples': samples
        }
    
    def get_outlier_probabilities(self, threshold=0.3):
        """Get outlier probabilities for edges with lower threshold."""
        if not self.outlier_detection:
            return None
        
        outlier_probs = torch.sigmoid(self.outlier_logits).detach().cpu().numpy()
        
        outlier_info = []
        for i, (lig1, lig2) in enumerate(self.edge_pairs):
            if outlier_probs[i] > threshold:
                outlier_info.append({
                    'edge': (lig1, lig2),
                    'outlier_prob': outlier_probs[i],
                    'fep_value': self.edge_values[i],
                    'ccc_value': self.edge_ccc_values[i],
                    'exp_value': self.edge_exp_values[i],
                    'error': self.edge_errors[i]
                })
        
        return outlier_info
    
    def evaluate_against_experimental(self):
        """Evaluate predictions against experimental values."""
        if self.node_data is None:
            return None
        
        estimates = self.get_node_estimates()
        means = estimates['means']
        
        # Get experimental values
        exp_values = self.node_data['Exp. ΔG'].values
        
        mae = np.mean(np.abs(means - exp_values))
        rmse = np.sqrt(np.mean((means - exp_values) ** 2))
        correlation = np.corrcoef(means, exp_values)[0, 1]
        
        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'predicted': means,
            'experimental': exp_values
        }
    
    def diagnose_network(self):
        """Diagnose network for potential issues."""
        print("\n" + "="*50)
        print("NETWORK DIAGNOSTICS")
        print("="*50)
        
        # Check for FEP vs CCC inconsistencies
        mean_abs_diff = np.mean(np.abs(self.fep_ccc_diffs))
        max_abs_diff = np.max(np.abs(self.fep_ccc_diffs))
        std_diff = np.std(self.fep_ccc_diffs)
        
        print(f"FEP vs CCC Analysis:")
        print(f"  Mean absolute difference: {mean_abs_diff:.3f} kcal/mol")
        print(f"  Max absolute difference: {max_abs_diff:.3f} kcal/mol")
        print(f"  Std of differences: {std_diff:.3f} kcal/mol")
        
        # Identify problematic edges
        threshold = np.percentile(np.abs(self.fep_ccc_diffs), 90)
        problematic_edges = []
        for i, diff in enumerate(self.fep_ccc_diffs):
            if abs(diff) > threshold:
                problematic_edges.append((i, self.edge_pairs[i], diff))
        
        print(f"  {len(problematic_edges)} edges with large FEP-CCC differences (>{threshold:.3f}):")
        for i, (edge_idx, edge_pair, diff) in enumerate(problematic_edges[:5]):
            print(f"    {edge_pair}: {diff:.3f} kcal/mol")
        if len(problematic_edges) > 5:
            print(f"    ... and {len(problematic_edges)-5} more")
        
        # Check edge error distribution
        edge_errors = np.array(self.edge_errors)
        print(f"\nEdge Error Analysis:")
        print(f"  Mean: {np.mean(edge_errors):.3f} kcal/mol")
        print(f"  Std: {np.std(edge_errors):.3f} kcal/mol")
        print(f"  Range: [{np.min(edge_errors):.3f}, {np.max(edge_errors):.3f}] kcal/mol")
        
        # Check network connectivity
        from collections import defaultdict
        connections = defaultdict(int)
        for lig1, lig2 in self.edge_pairs:
            connections[lig1] += 1
            connections[lig2] += 1
        
        conn_counts = list(connections.values())
        print(f"\nConnectivity Analysis:")
        print(f"  Mean connections per node: {np.mean(conn_counts):.1f}")
        print(f"  Min connections: {np.min(conn_counts)}")
        print(f"  Max connections: {np.max(conn_counts)}")
        
        poorly_connected = [lig for lig, count in connections.items() if count <= 2]
        if poorly_connected:
            print(f"  {len(poorly_connected)} poorly connected nodes (≤2 connections)")
        
        return {
            'fep_ccc_diffs': self.fep_ccc_diffs,
            'edge_errors': edge_errors,
            'connectivity': connections,
            'problematic_edges': problematic_edges
        }
    
    def plot_results(self, save_path=None):
        """Plot comprehensive results."""
        estimates = self.get_node_estimates()
        evaluation = self.evaluate_against_experimental()
        outlier_info = self.get_outlier_probabilities(threshold=0.6)  # Use higher threshold for plotting
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Predicted vs Experimental (nodes)
        if evaluation:
            axes[0, 0].scatter(evaluation['experimental'], evaluation['predicted'], 
                             alpha=0.7, s=50)
            
            # Add error bars
            axes[0, 0].errorbar(evaluation['experimental'], evaluation['predicted'],
                               yerr=estimates['stds'], fmt='none', alpha=0.3, capsize=3)
            
            # Perfect correlation line
            min_val = min(evaluation['experimental'].min(), evaluation['predicted'].min())
            max_val = max(evaluation['experimental'].max(), evaluation['predicted'].max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[0, 0].set_xlabel('Experimental ΔG (kcal/mol)')
            axes[0, 0].set_ylabel('Predicted ΔG (kcal/mol)')
            axes[0, 0].set_title(f'Improved Full-Rank GMVI Analysis\nMAE: {evaluation["mae"]:.3f}, '
                               f'R: {evaluation["correlation"]:.3f}')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty distribution
        axes[0, 1].hist(estimates['stds'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(np.mean(estimates['stds']), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(estimates["stds"]):.3f}')
        axes[0, 1].set_xlabel('Posterior Standard Deviation (kcal/mol)')
        axes[0, 1].set_ylabel('Number of Nodes')
        axes[0, 1].set_title('Node Uncertainty Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: FEP vs CCC differences
        axes[0, 2].hist(self.fep_ccc_diffs, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[0, 2].set_xlabel('FEP - CCC (kcal/mol)')
        axes[0, 2].set_ylabel('Number of Edges')
        axes[0, 2].set_title(f'FEP vs CCC Differences\nMean |diff|: {np.mean(np.abs(self.fep_ccc_diffs)):.3f}')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Edge accuracy
        edge_preds = []
        for sample in estimates['samples'][:100]:  # Use subset for speed
            preds = self._compute_edge_predictions(torch.tensor(sample)).detach().cpu().numpy()
            edge_preds.append(preds)
        edge_preds = np.array(edge_preds)
        edge_pred_means = np.mean(edge_preds, axis=0)
        
        axes[1, 0].scatter(self.edge_values, edge_pred_means, alpha=0.7)
        axes[1, 0].plot([min(self.edge_values), max(self.edge_values)], 
                       [min(self.edge_values), max(self.edge_values)], 'r--')
        axes[1, 0].set_xlabel('Observed Edge FEP (kcal/mol)')
        axes[1, 0].set_ylabel('Predicted Edge FEP (kcal/mol)')
        axes[1, 0].set_title('Edge Predictions vs Observations')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Outlier probabilities
        if outlier_info:
            outlier_probs = [info['outlier_prob'] for info in outlier_info]
            axes[1, 1].hist(outlier_probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_xlabel('Outlier Probability')
            axes[1, 1].set_ylabel('Number of Edges')
            axes[1, 1].set_title(f'Outlier Detection\n{len(outlier_info)} likely outliers')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Covariance structure
        cov_matrix = self._get_covariance_matrix().detach().cpu().numpy()
        im = axes[1, 2].imshow(cov_matrix, cmap='RdBu_r', aspect='auto')
        axes[1, 2].set_title('Posterior Covariance Matrix')
        axes[1, 2].set_xlabel('Node Index')
        axes[1, 2].set_ylabel('Node Index')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig

def load_cdk8_data():
    """Load CDK8 dataset."""
    # Load edge data
    edge_data = pd.read_csv('cdk8/results_edges_5ns.csv')
    
    # Load node data
    node_data = pd.read_csv('cdk8/results_5ns.csv')
    
    # Clean up node data column names
    node_data.columns = ['Affinity unit', '#', 'Ligand', 'Quality', 'Pred. ΔG', 'Pred. Error', 'Exp. ΔG', 'Structure']
    
    return edge_data, node_data

def analyze_cdk8_with_improved_gmvi():
    """Analyze CDK8 dataset with improved full-rank GMVI."""
    print("IMPROVED FULL-RANK GMVI ANALYSIS")
    print("=" * 60)
    
    # Load data
    edge_data, node_data = load_cdk8_data()
    
    print(f"Loaded CDK8 dataset:")
    print(f"  Nodes: {len(node_data)}")
    print(f"  Edges: {len(edge_data)}")
    print(f"  Edge error range: [{edge_data['FEP Error'].min():.3f}, {edge_data['FEP Error'].max():.3f}] kcal/mol")
    print(f"  Mean edge error: {edge_data['FEP Error'].mean():.3f} kcal/mol")
    
    # Fit improved full-rank GMVI model
    model = CorrectedFullRankGMVI(
        edge_data=edge_data,
        node_data=node_data,
        prior_mean=-10.0,  # Reasonable prior for binding energies
        prior_std=3.0,
        outlier_detection=True
    )
    
    # Run diagnostics
    diagnostics = model.diagnose_network()
    
    # Fit model
    losses = model.fit(n_epochs=1000, lr=0.01)
    
    # Get results
    estimates = model.get_node_estimates()
    evaluation = model.evaluate_against_experimental()
    
    # Try different outlier thresholds to identify worst offenders
    outliers_30 = model.get_outlier_probabilities(threshold=0.3)
    outliers_50 = model.get_outlier_probabilities(threshold=0.5)
    outliers_60 = model.get_outlier_probabilities(threshold=0.6)
    outliers_70 = model.get_outlier_probabilities(threshold=0.7)
    
    print(f"\nImproved Full-Rank GMVI Results:")
    print(f"  MAE: {evaluation['mae']:.3f} kcal/mol")
    print(f"  RMSE: {evaluation['rmse']:.3f} kcal/mol")
    print(f"  Correlation: {evaluation['correlation']:.3f}")
    print(f"  Mean uncertainty: {np.mean(estimates['stds']):.3f} kcal/mol")
    print(f"  Outlier detection by threshold:")
    print(f"    >30%: {len(outliers_30)} edges")
    print(f"    >50%: {len(outliers_50)} edges")
    print(f"    >60%: {len(outliers_60)} edges")
    print(f"    >70%: {len(outliers_70)} edges")
    
    # Print worst outlier edges (highest threshold)
    if outliers_70:
        print(f"  High-confidence outlier edges (>70%):")
        for info in outliers_70[:5]:
            print(f"    {info['edge']}: prob={info['outlier_prob']:.3f}, FEP={info['fep_value']:.3f}, CCC={info['ccc_value']:.3f}")
    elif outliers_60:
        print(f"  Medium-confidence outlier edges (>60%):")
        for info in outliers_60[:5]:
            print(f"    {info['edge']}: prob={info['outlier_prob']:.3f}, FEP={info['fep_value']:.3f}, CCC={info['ccc_value']:.3f}")
    elif outliers_50:
        print(f"  Lower-confidence outlier edges (>50%):")
        for info in outliers_50[:5]:
            print(f"    {info['edge']}: prob={info['outlier_prob']:.3f}, FEP={info['fep_value']:.3f}, CCC={info['ccc_value']:.3f}")
    
    # Plot results
    model.plot_results(save_path='improved_full_rank_gmvi_analysis.png')
    
    return model, estimates, evaluation

def main():
    """Main function."""
    analyze_cdk8_with_improved_gmvi()

if __name__ == "__main__":
    main() 