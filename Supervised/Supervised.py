import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Setup
if not os.path.exists('results'): os.makedirs('results')
print("Loading dataset...")
df = pd.read_csv("CurrentPopulationSurvey.csv")
df = df.dropna(thresh=0.8 * len(df), axis=1).dropna()  # Clean data
print(f"Cleaned shape: {df.shape}")

# Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Prepare features and target
X = df.drop(columns=['sex'])
y = df['sex'].map({1: 0, 2: 1})  # 0=male, 1=female
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN without PCA
knn = KNeighborsClassifier(n_neighbors=12, n_jobs=-1)
start_time = time.time()
knn.fit(X_train_scaled, y_train)
train_time_no_pca = time.time() - start_time

start_time = time.time()
y_pred = knn.predict(X_test_scaled)
inference_time_no_pca = time.time() - start_time

# Performance metrics without PCA
acc_no_pca = accuracy_score(y_test, y_pred)
precision_no_pca = precision_score(y_test, y_pred)
recall_no_pca = recall_score(y_test, y_pred)
f1_no_pca = f1_score(y_test, y_pred)
print(f"No PCA - Acc: {acc_no_pca:.4f}, Time: {train_time_no_pca:.4f}s")

# Apply PCA
pca_components = 35
pca = PCA(n_components=pca_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
explained_var = pca.explained_variance_ratio_
total_var_explained = sum(explained_var)

# KNN with PCA
knn_pca = KNeighborsClassifier(n_neighbors=12, n_jobs=-1)
start_time = time.time()
knn_pca.fit(X_train_pca, y_train)
train_time_pca = time.time() - start_time

start_time = time.time()
y_pred_pca = knn_pca.predict(X_test_pca)
inference_time_pca = time.time() - start_time

# Performance metrics with PCA
acc_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca)
recall_pca = recall_score(y_test, y_pred_pca)
f1_pca = f1_score(y_test, y_pred_pca)
print(f"With PCA - Acc: {acc_pca:.4f}, Time: {train_time_pca:.4f}s")

# Test on new samples
new_samples = X.sample(n=20, random_state=42)
true_genders = y[new_samples.index]
new_scaled = scaler.transform(new_samples)
start_time = time.time()
predictions_no_pca = knn.predict(new_scaled)
new_inference_time_no_pca = time.time() - start_time

new_pca = pca.transform(new_scaled)
start_time = time.time()
predictions_pca = knn_pca.predict(new_pca)
new_inference_time_pca = time.time() - start_time

new_acc_no_pca = accuracy_score(true_genders, predictions_no_pca)
new_acc_pca = accuracy_score(true_genders, predictions_pca)

# Results comparison
comparison_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
               'Training Time (sec)', 'Inference Time (sec)',
               'New Data Accuracy', 'New Data Inference Time (sec)'],
    'Without PCA': [acc_no_pca, precision_no_pca, recall_no_pca, f1_no_pca, 
                   train_time_no_pca, inference_time_no_pca,
                   new_acc_no_pca, new_inference_time_no_pca],
    'With PCA': [acc_pca, precision_pca, recall_pca, f1_pca, 
                train_time_pca, inference_time_pca,
                new_acc_pca, new_inference_time_pca]
}
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("results/performance_comparison.csv", index=False)

# Add visualizations
# 1. Confusion matrices
plt.figure(figsize=(12, 5))

# Confusion matrix without PCA
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Male', 'Female'])
disp.plot(ax=plt.gca(), cmap='Blues', values_format='d')
plt.title("Without PCA (k=12)")

# Confusion matrix with PCA
plt.subplot(1, 2, 2)
cm_pca = confusion_matrix(y_test, y_pred_pca)
disp_pca = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=['Male', 'Female'])
disp_pca.plot(ax=plt.gca(), cmap='Blues', values_format='d')
plt.title("With PCA (k=12)")

plt.tight_layout()
plt.savefig("results/confusion_matrices.png")
plt.close()

# 2. PCA variance explained
cumulative_var = np.cumsum(explained_var)
plt.figure(figsize=(10, 6))
plt.bar(range(1, pca_components + 1), explained_var, alpha=0.7, label='Individual')
plt.plot(range(1, pca_components + 1), cumulative_var, 'ro-', label='Cumulative')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Threshold')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Variance Explained')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/pca_variance.png")
plt.close()

# 3. Performance metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_data = comparison_df[comparison_df['Metric'].isin(metrics)]
plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, metrics_data['Without PCA'], width, label='Without PCA')
plt.bar(x + width/2, metrics_data['With PCA'], width, label='With PCA')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Performance Metrics Comparison')
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.savefig("results/performance_comparison.png")
plt.close()

# Final summary
print("\nFinal Summary:")
print(f"Dimension reduction: {(1 - pca_components/X_train_scaled.shape[1]) * 100:.2f}%")
print(f"Variance explained: {total_var_explained * 100:.2f}%")
print(f"Accuracy without PCA: {acc_no_pca:.4f}, with PCA: {acc_pca:.4f}")
print(f"Inference time improvement: {(1 - inference_time_pca/inference_time_no_pca) * 100:.2f}%")