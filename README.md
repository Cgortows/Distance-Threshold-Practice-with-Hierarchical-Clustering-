# Distance Threshold Practice with Hierarchical Clustering 

## Introduction
While I was looking into learning more about Customer Segmentation. I discovered that the random datasets I could find on the internet were not that great for visualizng clusters the way I wanted to. I decided to try working with synthetic data and it drove me to a better understanding of clustering, especially Hierarchical Clustersing. In order to visualize the datasets I will be using principal component analysis or PCA. I will use two components to keep it in two dimensions.

## Synthetic Datasets
We will start with the default perameters of the make_classification method from sklearn.datasets. The resulting dataset is not too impressive so we will make to make some adjustments. 
```
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_classification

# Set a seed for reproducibility
np.random.seed(42)

# Create a synthetic dataset
X, _ = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_clusters_per_class=2, n_classes=2)

# Convert to dataframe
df_complex = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])

# Initialize PCA with 2 components, Fit, Transform, and convert to a DataFrame
pca_complex = PCA(n_components=2)
pca_transformed_complex = pca_complex.fit_transform(df_complex)
df_pca_complex = pd.DataFrame(pca_transformed_complex, columns=['PC1', 'PC2'])

# Perform hierarchical clustering on PCA transformed data
Z_pca_complex = linkage(df_pca_complex)

# Using a distance threshold to define the clusters
clusters = fcluster(Z_pca_complex, 0.6, criterion='distance')

# Scatter plot of the hierarchical clustering results with discrete colormap
plt.figure(figsize=(8, 6))
plt.scatter(df_pca_complex['PC1'], df_pca_complex['PC2'], c=clusters, cmap='tab10', s=30, alpha=0.5)
plt.title('Default Settings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.savefig('my_plot.png')
plt.show()
```

![Default settings]()



