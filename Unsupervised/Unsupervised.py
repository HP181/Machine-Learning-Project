import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score

# Setup
import os
if not os.path.exists('kmeans_results'):
    os.makedirs('kmeans_results')

# 1. Load and preprocess data
print("Loading dataset...")
df = pd.read_csv("CurrentPopulationSurvey.csv")
df = df.dropna(thresh=0.8 * len(df), axis=1).dropna()
print(f"Cleaned shape: {df.shape}")

# Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 2. Prepare data
X = df.drop(columns=['sex'])
true_labels = df['sex'].map({1: 0, 2: 1})  # 0=male, 1=female
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply KMeans without PCA
k = 2  # Two genders
print(f"Applying KMeans (k={k}) without PCA...")
start_time = time.time()
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_
clustering_time = time.time() - start_time

# 4. Calculate metrics (without PCA)
print("Calculating metrics without PCA...")
sample_size = 10000
sample_indices = np.random.choice(X_scaled.shape[0], min(sample_size, X_scaled.shape[0]), replace=False)
silhouette = silhouette_score(X_scaled[sample_indices], cluster_labels[sample_indices])
ari = adjusted_rand_score(true_labels, cluster_labels)
homogeneity = homogeneity_score(true_labels, cluster_labels)
completeness = completeness_score(true_labels, cluster_labels)
print(f"No PCA - Silhouette: {silhouette:.4f}, ARI: {ari:.4f}")

# 5. Apply PCA
print("Applying PCA...")
pca_components = 35
pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(X_scaled)
total_var_explained = sum(pca.explained_variance_ratio_)
print(f"Variance explained: {total_var_explained*100:.2f}%")

# 6. Apply KMeans with PCA
print(f"Applying KMeans (k={k}) with PCA...")
start_time = time.time()
kmeans_pca = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
kmeans_pca.fit(X_pca)
cluster_labels_pca = kmeans_pca.labels_
clustering_time_pca = time.time() - start_time

# 7. Calculate metrics (with PCA)
print("Calculating metrics with PCA...")
silhouette_pca = silhouette_score(X_pca[sample_indices], cluster_labels_pca[sample_indices])
ari_pca = adjusted_rand_score(true_labels, cluster_labels_pca)
homogeneity_pca = homogeneity_score(true_labels, cluster_labels_pca)
completeness_pca = completeness_score(true_labels, cluster_labels_pca)
print(f"With PCA - Silhouette: {silhouette_pca:.4f}, ARI: {ari_pca:.4f}")

# 8. Cluster visualization
print("Creating visualizations...")
# Sample data for visualization
viz_sample_size = 10000
viz_sample_idx = np.random.choice(X_scaled.shape[0], min(viz_sample_size, X_scaled.shape[0]), replace=False)
true_labels_array = true_labels.values if hasattr(true_labels, 'values') else np.array(true_labels)

# Create 2D PCA visualization
pca_viz = PCA(n_components=2)
X_viz = pca_viz.fit_transform(X_scaled[viz_sample_idx])

# Cluster visualizations
plt.figure(figsize=(15, 5))

# No PCA clusters
plt.subplot(1, 3, 1)
plt.scatter(X_viz[:, 0], X_viz[:, 1], c=cluster_labels[viz_sample_idx], cmap='viridis', alpha=0.6, s=5)
plt.colorbar(label='Cluster')
plt.title(f'KMeans Clusters (No PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# PCA clusters
plt.subplot(1, 3, 2)
plt.scatter(X_viz[:, 0], X_viz[:, 1], c=cluster_labels_pca[viz_sample_idx], cmap='viridis', alpha=0.6, s=5)
plt.colorbar(label='Cluster')
plt.title(f'KMeans Clusters (With PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# True gender labels
plt.subplot(1, 3, 3)
plt.scatter(X_viz[:, 0], X_viz[:, 1], c=true_labels_array[viz_sample_idx], cmap='viridis', alpha=0.6, s=5)
plt.colorbar(label='True Gender (0=Male, 1=Female)')
plt.title('True Gender Labels')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.savefig("kmeans_results/cluster_visualization.png", dpi=300)
plt.close()

# 9. Cluster distribution heatmaps
plt.figure(figsize=(15, 6))

# Without PCA
plt.subplot(1, 2, 1)
sns.heatmap(
    pd.crosstab(
        pd.Series(cluster_labels, name='Cluster'),
        pd.Series(true_labels, name='Gender')
    ),
    annot=True, fmt='d', cmap="YlGnBu"
)
plt.title('Cluster vs Gender (No PCA)')

# With PCA
plt.subplot(1, 2, 2)
sns.heatmap(
    pd.crosstab(
        pd.Series(cluster_labels_pca, name='Cluster'),
        pd.Series(true_labels, name='Gender')
    ),
    annot=True, fmt='d', cmap="YlGnBu"
)
plt.title('Cluster vs Gender (With PCA)')

plt.tight_layout()
plt.savefig("kmeans_results/cluster_gender_heatmaps.png")
plt.close()

# 10. Performance metrics comparison
metrics = ['Silhouette Score', 'Adjusted Rand Score', 'Homogeneity', 'Completeness']
values_no_pca = [silhouette, ari, homogeneity, completeness]
values_pca = [silhouette_pca, ari_pca, homogeneity_pca, completeness_pca]

plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, values_no_pca, width, label='Without PCA')
plt.bar(x + width/2, values_pca, width, label='With PCA')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Clustering Quality Metrics Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.savefig("kmeans_results/clustering_metrics_comparison.png")
plt.close()

# 11. Final summary
print("\nFinal Summary:")
print(f"Dimension reduction: {(1 - pca_components/X_scaled.shape[1]) * 100:.2f}%")
print(f"Variance explained: {total_var_explained * 100:.2f}%")
print(f"Silhouette without PCA: {silhouette:.4f}, with PCA: {silhouette_pca:.4f}")
print(f"ARI without PCA: {ari:.4f}, with PCA: {ari_pca:.4f}")
print(f"Clustering time improvement: {(1 - clustering_time_pca/clustering_time) * 100:.2f}%")
print("All results saved to 'kmeans_results' directory")