import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
import hdbscan

# Load the LFW dataset
lfw = fetch_lfw_people(min_faces_per_person=50, resize=0.4)

print(f"Dataset size: {len(lfw.images)} images")
print(f"Image shape: {lfw.images[0].shape}")
print(f"Number of labels: {len(lfw.target_names)}")

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(lfw.images[i], cmap='gray')
    ax.set_title(lfw.target_names[lfw.target[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()


# Flatten the images
X = lfw.data
print(f"Data shape after flattening: {X.shape}")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"Reduced dimensions: {X_pca.shape}")

# Visualize explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()



# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
labels = clusterer.fit_predict(X_pca)

# Display cluster results
print(f"Number of clusters: {len(np.unique(labels))}")
print(f"Cluster labels: {np.unique(labels)}")

# Visualize clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Spectral', s=10)
plt.colorbar(label='Cluster Label')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("HDBSCAN Clustering")
plt.show()

noise = np.random.normal(0, 0.1, X_scaled.shape)
X_noisy = X_scaled + noise
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='manhattan')
labels = clusterer.fit_predict(X_pca)
from sklearn.metrics import silhouette_score

# Silhouette score (excluding noise points)
filtered_data = X_pca[labels != -1]
filtered_labels = labels[labels != -1]

if len(np.unique(filtered_labels)) > 1:
    score = silhouette_score(filtered_data, filtered_labels)
    print(f"Silhouette Score: {score}")
else:
    print("Not enough clusters for silhouette score.")
new_image = X_pca[0].reshape(1, -1)
cluster_label = clusterer.single_linkage_tree_.get_clusters(new_image, clusterer.min_cluster_size)
print(f"Assigned cluster: {cluster_label}")

representative_indices = clusterer.exemplars_
for idx in representative_indices:
    plt.imshow(lfw.images[idx], cmap='gray')
    plt.title(f"Cluster {labels[idx]}")
    plt.show()
