#!/usr/bin/env python3
"""
Percentile-based threshold detection using FAISS KDE
For anomaly detection without labels - finds threshold based on density percentiles
"""

import numpy as np
import torch
import faiss
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_env_data, ReplayBufferAbiomed



class PercentileThresholdKDE:
    """
    KDE-based anomaly detection using percentile thresholds
    """
    def __init__(self, bandwidth=1.0, n_neighbors=100, use_gpu=True, 
                 normalize=True, percentile=5.0, devid=0):
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.normalize = normalize
        self.percentile = percentile  # e.g., 5% for bottom 5% as anomalies
        
        self.index = None
        self.training_data = None
        self.scaler = None
        self.threshold = None
        self.is_fitted = False
        self.devid = devid
        
    def fit(self, X, verbose=True):
        """
        Fit the model and compute percentile-based threshold
        
        Args:
            X: Training data (n_samples, n_features)
            verbose: Print fitting information
        """
        start_time = time.time()
        
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        
        X = X.astype(np.float32)
        original_X = X.copy()  # Keep for threshold computation

        
        # Normalize data if requested
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        #FIT PCA AND TRANSFORM DATA
        if X.shape[1] > 17:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=7)
            self.pca.fit(X)
            X = self.pca.transform(X)
            if verbose:
                print(f"Reduced to {X.shape[1]} features using PCA")
        
        
        self.training_data = X.copy()
        n_samples, n_features = X.shape
        
        if verbose:
            print(f"Training on {n_samples} samples with {n_features} features")
            print(f"Percentile threshold: {self.percentile}%")
            print(f"Bandwidth: {self.bandwidth}, K-neighbors: {self.n_neighbors}")
        
        # Build FAISS index
        if n_features <= 64 and n_samples < 1000000:
            self.index = faiss.IndexFlatL2(n_features)
        else:
            nlist = min(int(np.sqrt(n_samples)), 4096)
            quantizer = faiss.IndexFlatL2(n_features)
            self.index = faiss.IndexIVFFlat(quantizer, n_features, nlist)
            if hasattr(self.index, 'train'):
                self.index.train(X)
        
        # Move to GPU if available
        if self.use_gpu:
            try:
                gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resources, self.devid, self.index)
                if verbose:
                    print("Using GPU acceleration")
            except Exception as e:
                if verbose:
                    print(f"GPU failed, using CPU: {e}")
                self.use_gpu = False
        
        self.index.add(X)
        
        # Compute density scores on training data to find threshold
        if verbose:
            print("Computing density scores for threshold...")
        
        density_scores = self._score_samples_internal(X)
        
        # Find percentile threshold (lower density = higher anomaly score)
        self.threshold = np.percentile(density_scores, self.percentile)
        
        self.is_fitted = True
        fit_time = time.time() - start_time
        
        if verbose:
            print(f"Fitting completed in {fit_time:.2f} seconds")
            print(f"Threshold (log-density): {self.threshold:.4f}")
            print(f"Expected anomaly rate: {self.percentile}%")
        
        return self
    
    def _score_samples_internal(self, X):
        """Internal method to compute density scores"""
        k = min(self.n_neighbors, self.training_data.shape[0])
        distances, indices = self.index.search(X, k)
        
        # Gaussian kernel
        kernel_values = np.exp(-0.5 * distances / (self.bandwidth**2))
        density = np.mean(kernel_values, axis=1)
        log_density = np.log(density + 1e-10)
        
        return log_density
    
    def score_samples(self, X):
        """
        Compute log-density estimates for new data
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            log_density: Log-density estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        
        X = X.astype(np.float32)
        
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        
        if self.pca is not None:
            X = self.pca.transform(X)
        
        return self._score_samples_internal(X)
    
    def predict(self, X):
        """
        Predict anomalies based on threshold
        
        Args:
            X: Test data
            
        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold, 1, -1)
    
    def decision_function(self, X):
        """
        Anomaly score (higher = more anomalous)
        
        Args:
            X: Test data
            
        Returns:
            anomaly_scores: Higher values indicate more anomalous
        """
        density_scores = self.score_samples(X)
        # Convert to anomaly scores (negative log-density relative to threshold)
        return self.threshold - density_scores
    
    def get_threshold_stats(self):
        """Get statistics about the threshold"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return {
            'threshold': self.threshold,
            'percentile': self.percentile,
            'bandwidth': self.bandwidth,
            'n_neighbors': self.n_neighbors,
            'n_training_samples': self.training_data.shape[0]
        }
    def save_model(self, base_path):
        """Quick save function - saves FAISS index and metadata separately"""
        
        # if self.use_gpu and hasattr(self.index, 'index'):
            # For GPU index
        print("Transfering from gpu to cpu for saving...")
        index_cpu = faiss.index_gpu_to_cpu(self.index)
        # else:
        #     index_cpu = self.index
        
        faiss.write_index(index_cpu, f"{base_path}.faiss")
        
        # Save metadata
        metadata = {
            'threshold': self.threshold,
            'bandwidth': self.bandwidth,
            'n_neighbors': self.n_neighbors,
            'percentile': self.percentile,
            'normalize': self.normalize,
            'scaler': self.scaler,
            'training_data': self.training_data,
            'is_fitted': self.is_fitted,
            'model_params': self.get_threshold_stats(),
            'pca': self.pca if hasattr(self, 'pca') else None,
        }
        
        with open(f"{base_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Model saved to {base_path}.faiss and {base_path}_metadata.pkl")

    @classmethod
    def load_model(cls, load_path, use_gpu=True, devid=0):
        """
        Load a saved model
        
        Args:
            load_path: Base path for loading (without extension)
            load_format: 'auto', 'separate', or 'json_meta' (combined not supported)
            use_gpu: Whether to move index to GPU after loading
            devid: GPU device ID
        
        Returns:
            Loaded PercentileThresholdKDE model
        """
    
        # Load FAISS index
        index = faiss.read_index(f"{load_path}.faiss")
        
        # Load metadata
        with open(f"{load_path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create model instance
        model = cls(
            bandwidth=metadata['bandwidth'],
            n_neighbors=metadata['n_neighbors'],
            use_gpu=use_gpu,
            normalize=metadata['normalize'],
            percentile=metadata['percentile'],
            devid=devid
        )
        
        # Restore state
        # model['model'] = index
        model.index = index
        model.threshold = metadata['threshold']
        model.scaler = metadata['scaler']
        model.training_data = metadata['training_data']
        model.is_fitted = metadata['is_fitted']
        model.pca = metadata['pca'] if 'pca' in metadata else None
        
        
        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            try:
                gpu_resources = faiss.StandardGpuResources()
                model.index = faiss.index_cpu_to_gpu(gpu_resources, devid, model.index)
                model.use_gpu = True
                print(f"Model loaded and moved to GPU {devid}")
            except Exception as e:
                print(f"Could not move to GPU: {e}, using CPU")
                model.use_gpu = False
        else:
            model.use_gpu = False

        model_dict  = {'model': model,'model_index': model.index, 'thr': model.threshold, 'scaler': metadata['scaler'],}
        print(f"Model loaded from {load_path}")
        return model_dict

def load_data(data_path, test_size, validation_size, args=None):
    """
    Load data from various file formats
    
    Args:
        data_path: Path to data file
        file_format: 'csv', 'npy', 'npz', 'pkl', 'txt', or 'auto' (detect from extension)
        columns: List of column indices or names to use (None = use all)
        delimiter: Delimiter for CSV/txt files
        
    Returns:
        X: Data array (n_samples, n_features)
    """
    # data_path = "/abiomed/intermediate_data_d4rl/farama_sac_expert/Hopper-v2_expert_1000.pkl"
    
    
    print(f"Loading data from {data_path}")
    if args.env == "abiomed":
        env, data = get_env_data(args)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] 

        replay_buffer = ReplayBufferAbiomed(state_dim, action_dim)
        replay_buffer.convert_abiomed(data, env)
        X = np.concatenate([replay_buffer.state, replay_buffer.action], axis=1)
    else:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        X = np.concatenate([data['observations'], data['actions']], axis=1)
   
    n_samples = len(X)
    
    
    # Sequential split for time series data
    test_end = int(n_samples * (1 - test_size))
    val_end = int(test_end * (1 - validation_size))
    
    train_idx = np.arange(0, val_end)
    val_idx = np.arange(val_end, test_end)
    test_idx = np.arange(test_end, n_samples) 
    
    
    # Ensure 2D array
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_train = X[train_idx]
    X_val = X[val_idx] if len(val_idx) > 0 else None
    X_test = X[test_idx]
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'split_info': {
            'total_samples': n_samples,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'test_samples': len(test_idx),
        }
    }




def evaluate_anomaly_detection(model, X_test, y_true, verbose=True):
    """
    Evaluate anomaly detection performance
    
    Args:
        model: Trained anomaly detection model
        X_test: Test data
        y_true: True labels (1=normal, -1=anomaly)
        verbose: Print results
    """
    # Get predictions and scores
    y_pred = model.predict(X_test)
    anomaly_scores = model.decision_function(X_test)
    
    # Compute metrics
    tp = np.sum((y_true == -1) & (y_pred == -1))  # True anomalies detected
    fp = np.sum((y_true == 1) & (y_pred == -1))   # False alarms
    tn = np.sum((y_true == 1) & (y_pred == 1))    # True normals
    fn = np.sum((y_true == -1) & (y_pred == 1))   # Missed anomalies
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    accuracy = (tp + tn) / len(y_true)
    
    if verbose:
        print(f"\n=== Anomaly Detection Results ===")
        print(f"True anomalies in test set: {np.sum(y_true == -1)}")
        print(f"Predicted anomalies: {np.sum(y_pred == -1)}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Confusion Matrix:")
        print(f"  TP: {tp:4d} | FP: {fp:4d}")
        print(f"  FN: {fn:4d} | TN: {tn:4d}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }


def plot_anomaly_results(X, y_true, y_pred, anomaly_scores, save_path=None):
    """
    Plot anomaly detection results (works for 2D data)
    """
    if X.shape[1] != 2:
        print("Plotting only available for 2D data")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: True labels
    normal_mask = y_true == 1
    anomaly_mask = y_true == -1
    
    axes[0].scatter(X[normal_mask, 0], X[normal_mask, 1], 
                   c='blue', alpha=0.6, s=20, label='Normal')
    axes[0].scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                   c='red', alpha=0.8, s=20, label='True Anomalies')
    axes[0].set_title('True Labels')
    axes[0].legend()
    
    # Plot 2: Predictions
    pred_normal_mask = y_pred == 1
    pred_anomaly_mask = y_pred == -1
    
    axes[1].scatter(X[pred_normal_mask, 0], X[pred_normal_mask, 1], 
                   c='blue', alpha=0.6, s=20, label='Predicted Normal')
    axes[1].scatter(X[pred_anomaly_mask, 0], X[pred_anomaly_mask, 1], 
                   c='red', alpha=0.8, s=20, label='Predicted Anomalies')
    axes[1].set_title('Predictions')
    axes[1].legend()
    
    # Plot 3: Anomaly scores
    scatter = axes[2].scatter(X[:, 0], X[:, 1], c=anomaly_scores, 
                            cmap='viridis', alpha=0.7, s=20)
    axes[2].set_title('Anomaly Scores')
    plt.colorbar(scatter, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def find_optimal_percentile(X_train, X_val=None, percentiles=None, bandwidth=1.0, 
                           n_neighbors=100, use_gpu=True, metric='density_range'):
    """
    Find optimal percentile threshold using unsupervised criteria (no labels needed)
    
    Args:
        X_train: Training data
        X_val: Validation data (if None, uses part of training data)
        percentiles: List of percentiles to test
        bandwidth: KDE bandwidth
        n_neighbors: Number of neighbors for FAISS
        use_gpu: Use GPU acceleration
        metric: 'density_range', 'stability', or 'separation'
    """
    if percentiles is None:
        percentiles = [1, 2, 3, 5, 7, 10, 15, 20]
    
    print(f"\n=== Finding Optimal Percentile (Unsupervised) ===")
    print(f"Testing percentiles: {percentiles}")
    print(f"Optimization metric: {metric}")
    
    if X_val is None:
        # Use 20% of training data for validation
        n_val = int(0.2 * len(X_train))
        indices = np.random.permutation(len(X_train))
        X_train_subset = X_train[indices[n_val:]]
        X_val = X_train[indices[:n_val]]
        print(f"Created validation set: {len(X_val)} samples")
    else:
        X_train_subset = X_train
    
    best_score = -np.inf
    best_percentile = percentiles[0]
    results = []
    
    for p in percentiles:
        print(f"Testing percentile: {p}%")
        
        model = PercentileThresholdKDE(
            bandwidth=bandwidth,
            n_neighbors=n_neighbors,
            use_gpu=use_gpu,
            percentile=p
        )
        
        model.fit(X_train_subset, verbose=False)
        
        # Get density scores for validation data
        val_scores = model.score_samples(X_val)
        val_predictions = model.predict(X_val)
        
        # Compute unsupervised quality metrics
        if metric == 'density_range':
            # Maximize the range of density scores (better separation)
            score = np.max(val_scores) - np.min(val_scores)
            
        elif metric == 'stability':
            # Minimize variance in "normal" region (more stable threshold)
            normal_scores = val_scores[val_predictions == 1]
            if len(normal_scores) > 0:
                score = -np.var(normal_scores)  # Negative because we want low variance
            else:
                score = -np.inf
                
        elif metric == 'separation':
            # Maximize separation between normal and anomalous regions
            normal_scores = val_scores[val_predictions == 1]
            anomaly_scores = val_scores[val_predictions == -1]
            
            if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                normal_mean = np.mean(normal_scores)
                anomaly_mean = np.mean(anomaly_scores)
                pooled_std = np.sqrt((np.var(normal_scores) + np.var(anomaly_scores)) / 2)
                
                # Cohen's d (effect size)
                score = abs(normal_mean - anomaly_mean) / (pooled_std + 1e-10)
            else:
                score = -np.inf
        
        anomaly_rate = (val_predictions == -1).mean()
        
        results.append({
            'percentile': p,
            'score': score,
            'anomaly_rate': anomaly_rate,
            'threshold': model.threshold,
            'val_score_mean': np.mean(val_scores),
            'val_score_std': np.std(val_scores)
        })
        
        print(f"  Score: {score:.4f}, Anomaly rate: {anomaly_rate:.1%}, "
              f"Threshold: {model.threshold:.4f}")
        
        if score > best_score:
            best_score = score
            best_percentile = p
    
    print(f"\nBest percentile: {best_percentile}% (Score: {best_score:.4f})")
    return best_percentile, results



def main():
    parser = argparse.ArgumentParser(description='Percentile-based KDE anomaly detection')
    
    # Data loading arguments
    parser.add_argument('--data_path', type=str, default = "/abiomed/intermediate_data_d4rl/farama_sac_expert/Hopper-v2_expert_1000.pkl",help='Path to data file')

    # Data splitting arguments
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set fraction')
    parser.add_argument('--temporal_split', action='store_true', 
                       help='Use temporal split (no shuffle) for time series')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    
    # Model arguments
    parser.add_argument('--percentile', type=float, default=5.0, help='Percentile threshold')
    parser.add_argument('--bandwidth', type=float, default=1.0, help='KDE bandwidth')
    parser.add_argument('--k_neighbors', type=int, default=100, help='Number of neighbors')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--no_normalize', action='store_true', help='Disable data normalization')
    
    # Optimization arguments
    parser.add_argument('--optimize_percentile', action='store_true', 
                       help='Find optimal percentile')
    parser.add_argument('--optimization_metric', choices=['density_range', 'stability', 'separation'],
                       default='density_range', help='Metric for percentile optimization')
    
    # Output arguments
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save_model', type=str, default="trained_kde", help='Save model path')
    parser.add_argument('--save_results', type=str, help='Save results to file')
    parser.add_argument('--devid', type=int, default=0, help='GPU device ID (if using GPU)')
    parser.add_argument('--save_path', type=str, default='/abiomed/models/kde', help='Path to save model and results')
    
    # Synthetic data fallback (if no data_path provided)
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of synthetic samples')
    parser.add_argument('--n_features', type=int, default=2, help='Number of synthetic features')
    parser.add_argument('--anomaly_rate', type=float, default=0.05, help='Synthetic anomaly rate')

    # noisy dataset parameters
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")
    parser.add_argument("--env", type=str, default="")

    #============ abiomed environment arguments ============
    parser.add_argument("--model_name", type=str, default="10min_1hr_window")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--action_space_type", type=str, default="continuous", choices=["continuous", "discrete"], help="Type of action space for the environment") 

    args = parser.parse_args()
    
    print("=== Percentile-Based KDE Anomaly Detection ===")
    
    # Load data
    print(f"\nLoading data from: {args.data_path}")
    
    
    print(f"\nSplitting data...")
    data_splits = load_data(args.data_path, test_size=args.test_size, validation_size=args.val_size, args=args)

    X_train = data_splits['X_train']
    X_val = data_splits['X_val']
    X_test = data_splits['X_test']
    
    # No true labels available
    y_test = None
    
    print(f"\nFinal data splits:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape if X_val is not None else 'None'}")
    print(f"  Test: {X_test.shape}")
    
    # Find optimal percentile if requested
    if args.optimize_percentile:
        print(f"\nOptimizing percentile using '{args.optimization_metric}' metric...")
        optimal_percentile, search_results = find_optimal_percentile(
            X_train, X_val,
            bandwidth=args.bandwidth,
            n_neighbors=args.k_neighbors,
            use_gpu=not args.no_gpu,
            metric=args.optimization_metric
        )
        percentile_to_use = optimal_percentile
    else:
        percentile_to_use = args.percentile
    
    # Train final model
    print(f"\nTraining final model with {percentile_to_use}% threshold...")
    model = PercentileThresholdKDE(
        bandwidth=args.bandwidth,
        n_neighbors=args.k_neighbors,
        use_gpu=not args.no_gpu,
        normalize=not args.no_normalize,
        percentile=percentile_to_use,
        devid = args.devid
    )
    
    start_time = time.time()
    model.fit(X_train)
    fit_time = time.time() - start_time
    
    print(f"Training completed in {fit_time:.2f} seconds")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_predictions = model.predict(X_test)
    test_scores = model.score_samples(X_test)
    test_anomaly_scores = model.decision_function(X_test)
    
    anomaly_count = np.sum(test_predictions == -1)
    anomaly_rate = anomaly_count / len(X_test)
    
    print(f"Test set anomalies detected: {anomaly_count}/{len(X_test)} ({anomaly_rate:.1%})")
    print(f"Density score range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")
    print(f"Anomaly score range: [{test_anomaly_scores.min():.4f}, {test_anomaly_scores.max():.4f}]")
    

    # Generate plots
    if args.plot and X_test.shape[1] == 2:
        print("Generating plots...")
        if y_test is not None:
            plot_anomaly_results(X_test, y_test, test_predictions, test_anomaly_scores, 
                               'anomaly_detection_results.png')
        else:
            # Plot without true labels
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.scatter(X_test[:, 0], X_test[:, 1], c=test_predictions, 
                       cmap='coolwarm', alpha=0.6, s=20)
            plt.title('Predicted Labels')
            plt.colorbar(label='Prediction')
            
            plt.subplot(1, 2, 2)
            plt.scatter(X_test[:, 0], X_test[:, 1], c=test_anomaly_scores, 
                       cmap='viridis', alpha=0.6, s=20)
            plt.title('Anomaly Scores')
            plt.colorbar(label='Anomaly Score')
            
            plt.tight_layout()
            plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Plot saved to anomaly_detection_results.png")
    
    # Save model
    # if args.save_model:
    #     print(f"Saving model to {args.save_model}...")
    #     index_cpu = faiss.index_gpu_to_cpu(model.index)
    #     faiss.write_index(index_cpu, f'{args.save_model}.faiss')
    #    
    #     print("Model saved!")
    suffix_parts = [args.save_model]
    if args.action:
        suffix_parts.append(f"action")
    if args.transition:
        suffix_parts.append(f"obs")
    save_path = "_".join(suffix_parts)
    if args.save_model:
        model.save_model(os.path.join(args.save_path, save_path)) 
    
    # Save results
    if args.save_results:
        results = {
            'model_params': model.get_threshold_stats(),
            'data_info': data_splits['split_info'],
            'test_predictions': test_predictions,
            'test_scores': test_scores,
            'test_anomaly_scores': test_anomaly_scores,
            'anomaly_rate': anomaly_rate,
            'fit_time': fit_time
        }
      
        if args.optimize_percentile:
            results['optimization_results'] = search_results
        
        with open(args.save_results, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {args.save_results}")
    
    # Print threshold statistics
    stats = model.get_threshold_stats()
    print(f"\n=== Model Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n=== Usage Example ===")
    print("# To use the trained model on new data:")
    print("scores = model.score_samples(new_data)")
    print("predictions = model.predict(new_data)  # 1=normal, -1=anomaly")  
    print("anomaly_scores = model.decision_function(new_data)  # higher=more anomalous")
    print(f"# Threshold: {model.threshold:.4f}")


if __name__ == "__main__":
    main()