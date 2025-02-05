# Face Clustering with HDBSCAN and PCA

## Overview
This project performs unsupervised clustering on facial images from the Labeled Faces in the Wild (LFW) dataset using HDBSCAN and PCA. The goal is to identify natural groupings of similar faces based on extracted features.

## Features & Steps
1. **Load Dataset:** Fetch the LFW dataset and display sample images.
2. **Preprocessing:** Flatten images and standardize the data.
3. **Dimensionality Reduction:** Apply Principal Component Analysis (PCA) to retain 95% variance.
4. **Clustering with HDBSCAN:** Identify clusters using HDBSCAN with different distance metrics.
5. **Evaluation:**
   - Visualize clusters in 2D PCA space.
   - Compute Silhouette Score to assess clustering quality.
   - Display representative cluster images.

## Dependencies
- numpy
- matplotlib
- scikit-learn
- hdbscan

## How to Run
1. Install dependencies:
   ```sh
   pip install numpy matplotlib scikit-learn hdbscan
   ```
2. Run the script:
   ```sh
   python clustering_faces.py
   ```
3. View the generated visualizations and cluster assignments.

## Results & Insights
- PCA effectively reduces dimensionality while preserving key features.
- HDBSCAN identifies meaningful clusters, while also recognizing noise (outliers).
- The Silhouette Score provides a quantitative measure of cluster quality.

## Future Enhancements
- Experiment with different distance metrics for clustering.
- Incorporate deep learning-based feature extraction for improved clustering.
- Explore interactive visualizations for cluster exploration.

---


