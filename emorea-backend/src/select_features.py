'''
 # @ Author: Sofia Condesso
 # @ Create Time: 2025-03-03
 # @ Description: Implementation of feature selection algorithms proposed by Ferreira, A., Figueiredo, M. (2012).
 #                - Algorithm 1: Relevance-only feature selection
 #                  - Dispersion measures: MAD, MM, TV, AMGM
 #                - Algorithm 2: Relevance-redundancy feature selection
 #                  - Similarity measures: AC, CC, MICI
 #                - Default parameters are set to the paper recommendations
 #                - Example usage is provided at the end of the script
 #
 # @ References: Artur J. Ferreira and MáRio A. T. Figueiredo. 2012. 
 #               Efficient feature selection filters for high-dimensional data. 
 #               Pattern Recogn. Lett. 33, 13 (October, 2012), 1794–1804. 
 #               https://doi.org/10.1016/j.patrec.2012.05.019
 '''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


def calculate_dispersion(X, measure):
    """
    Compute dispersion measures
    """
    n_samples, n_features = X.shape
    dispersions = np.zeros(n_features)
    
    for i in range(n_features):
        feature = X[:, i]
        
        if measure == 'MAD':
            # Mean Absolute Deviation from the mean
            mean = np.mean(feature)
            dispersions[i] = np.mean(np.abs(feature - mean))
            
        elif measure == 'MM':
            # Mean-Median difference
            mean = np.mean(feature)
            med = np.median(feature)
            dispersions[i] = np.abs(mean - med)
            
        elif measure == 'TV':
            # Sample variance (1/n)
            dispersions[i] = np.var(feature, ddof=0)
            
        elif measure == 'AMGM':
            # Arithmetic Mean / Geometric Mean of exponentials (stabilized)
            max_val = np.max(feature)
            exp_feature = np.exp(feature - max_val)  # Numerical stability
            arithmetic_mean = np.mean(exp_feature)
            geometric_mean = np.exp(np.mean(feature - max_val))  # Adjust for stabilization
            dispersions[i] = arithmetic_mean / geometric_mean
            
        else:
            raise ValueError(f"Unknown dispersion measure: {measure}")
    
    return dispersions

def calculate_similarity(Xi, Xj, measure):
    """
    Compute feature similarity measures
    """
    if measure == 'AC':
        # Absolute cosine similarity (vectors)
        dot = np.dot(Xi, Xj)
        norm_i = np.linalg.norm(Xi)
        norm_j = np.linalg.norm(Xj)
        if norm_i == 0 or norm_j == 0:
            return 0.0
        return np.abs(dot) / (norm_i * norm_j)
    
    elif measure == 'CC':
        # Pearson correlation coefficient
        mean_i = np.mean(Xi)
        mean_j = np.mean(Xj)
        cov = np.mean((Xi - mean_i) * (Xj - mean_j))
        std_i = np.std(Xi, ddof=0)
        std_j = np.std(Xj, ddof=0)
        if std_i == 0 or std_j == 0:
            return 0.0
        return np.abs(cov / (std_i * std_j))
    
    elif measure == 'MICI':
        # Maximal Information Compression Index
        var_i = np.var(Xi, ddof=0)
        var_j = np.var(Xj, ddof=0)
        cov = np.cov(Xi, Xj, ddof=0)[0, 1]
        lam = 0.5 * (var_i + var_j - np.sqrt((var_i - var_j)**2 + 4*cov**2))
        return 2 * lam
        
    else:
        raise ValueError(f"Unknown similarity measure: {measure}")


# Visualize mask
def plot_feature_mask(mask, title):
    n_features = mask.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_features)))
    pad = grid_size**2 - n_features
    
    mask_padded = np.pad(mask, (0, pad))
    plt.imshow(mask_padded.reshape(grid_size, grid_size), cmap='gray')
    plt.title(title)
    plt.show()



def algorithm1(
    X, 
    L: float = 0.9,  # Fraction of features to initially select (0.7-0.9)
    dispersion_measure: str = 'MM'  # Default to Mean-Median difference
):
    """
    Relevance-only feature selection with paper-recommended defaults
    
    Args:
        X: Input data matrix (n_samples, n_features)
        L: Fraction of features to select (0.7-0.9 per paper)
        dispersion_measure: Dispersion measure to use
        
    Returns:
        Selected feature indices
    """
    # Calculate number of features based on fraction L
    d = X.shape[1]
    m = int(L * d)

    # Paper recommendations
    if not 0.7 <= L <= 0.9:
        print(f"Warning: L={L:.2f} outside recommended [0.7, 0.9] range")

    dispersions = calculate_dispersion(X, dispersion_measure)
    sorted_indices = np.argsort(dispersions)[::-1]  # Descending order
    return sorted_indices[:m]

def algorithm2(
    X,
    L: float = 0.95,  # Fraction of features to initially consider (0.7-0.99)
    MS: float = 0.8,  # Similarity threshold (0.7-0.8)
    dispersion_measure: str = 'MM',  # Default to Mean-Median difference
    similarity_measure: str = 'AC'   # Absolute cosine similarity
):
    """
    Relevance-redundancy feature selection with optimal parameters
    
    Args:
        X: Input data matrix
        L: Fraction of top features to consider
        MS: Maximum allowed similarity
        
    Returns:
        Selected feature indices
    """
    # Calculate number of features based on fraction L
    d = X.shape[1]
    m = int(L * d)
    
    # Validate parameter ranges
    if not 0.7 <= L <= 0.99:
        print(f"Warning: L={L:.2f} outside recommended [0.7, 0.99] range")
    if not 0.7 <= MS <= 0.8:
        print(f"Warning: MS={MS:.2f} outside recommended [0.7, 0.8] range")

    # Sort features by dispersion
    dispersions = calculate_dispersion(X, dispersion_measure)
    sorted_indices = np.argsort(dispersions)[::-1]

    # Ensure that the most relevant feature, as determined by the dispersion measure, 
    # is always included in the final selected feature set.
    selected_features = [sorted_indices[0]]
    prev_feature = sorted_indices[0]
    
    # Iterate over the remaining features
    for current_feature in sorted_indices[1:]:
        if len(selected_features) >= m:
            break
            
        # Compute similarity between current feature and last selected feature
        similarity = calculate_similarity(
            X[:, current_feature], 
            X[:, prev_feature], 
            similarity_measure
        )
        
        if similarity < MS:
            selected_features.append(current_feature)
            prev_feature = current_feature
    
    return np.array(selected_features)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, algorithm='algorithm2', L=0.95, MS=0.8, dispersion_measure='MM'):
        self.algorithm = algorithm
        self.L = L
        self.MS = MS
        self.dispersion_measure = dispersion_measure
        self.selected_indices = None
        
    def fit(self, X, y=None):
        if self.algorithm == 'algorithm1':
            self.selected_indices = algorithm1(X, self.L, self.dispersion_measure)
        else:
            self.selected_indices = algorithm2(X, self.L, self.MS, self.dispersion_measure)
        return self
    
    def transform(self, X):
        return X[:, self.selected_indices]

if __name__ == "__main__":

    import numpy as np
    import time

    # Generate a random dataset with paper-like characteristics
    np.random.seed(42)
    X_data = np.random.randn(100, 50)  # 100 samples, 50 features
    y_labels = np.random.randint(0, 2, 100)  # Binary classification

    print("Data shape:", X_data.shape)

    # Example 1: Basic usage with paper recommendations
    print("\nRunning Algorithm 1 with default recommendations (L=0.9, MM):")
    start0 = time.time()
    selected_alg1 = algorithm1(X_data)
    end = time.time()
    print(f"Time taken: {end - start0:.4f} seconds")
    print(f"Selected {len(selected_alg1)} features (indices):\n{selected_alg1}")

    print("\nRunning Algorithm 2 with optimal pair (L=0.95, MS=0.8):")
    start = end
    selected_alg2 = algorithm2(X_data)
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    print(f"Selected {len(selected_alg2)} features (indices):\n{selected_alg2}")

    # Example 2: Custom parameters within recommended ranges
    print("\nCustom Algorithm 2 (L=0.85, MS=0.75, MAD):")
    start = end
    selected_custom = algorithm2(
        X_data,
        L=0.85,
        MS=0.75,
        dispersion_measure='MAD'
    )
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    print(f"Selected {len(selected_custom)} features (indices):\n{selected_custom}")

    # Example 3: Force parameters outside recommendations
    print("\nForced parameters (Algorithm 1 with L=0.95 - expect warning):")
    start = end
    selected_force = algorithm1(X_data, L=0.95)
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    print(f"Selected {len(selected_force)} features (indices):\n{selected_force}")

    print(f"Total time for all runs: {end - start0:.4f} seconds")


