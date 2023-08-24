# Distance Threshold Practice with Hierarchical Clustering 

While I was looking into learning more about Customer Segmentation. I discovered that the random datasets I could find on the internet were not that great for visualizng clusters the way I wanted to. I decided to try working with synthetic data and it drove me to a better understanding of clustering, especially Hierarchical Clustersing. In order to visualize the datasets I will be using principal component analysis or PCA. I will use two components to keep it in two dimensions.

## Synthetic Datasets
We will start with the default perameters of the make_classification method from sklearn.datasets. The resulting dataset is not too impressive so we will have to make some adjustments. 
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
distance_threshold = 50
clusters = fcluster(Z_pca_complex, distance_threshold, criterion='distance')

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

![Default settings](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/default_make_classification_settings.png)

Since most datasets will have far more than 100 samples, we should increse the n_samples perameter from 100 to 1000 to get a more realistic dataset.

![1000_n_samples](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/1000_samples.png)

  With the larger dataset, the n_clusters_per_class and n_classes perameters become more apparent. The graph shows four or five relativly distinct clusters. This is where it comes down to how you choose to view your data. With the set perameters, n_clusters_per_class = 2 and n_classes = 2, we should have four clusters. But as we can see it could be four or it could be five. 
To better view the clusters lets try using the distance threshold perameter in the fcluster method. 

  First lets add method='ward' to the linkage function we were using. This perameter determines how the linkage distances between clusters are calculated during the process of merging clusters. Different linkage methods lead to different structures of hierarchical clusters. In the context of hierarchical clustering, a "linkage distance" is a measure of dissimilarity between two clusters. When combining clusters into larger ones, the linkage distance between them is used to determine how similar or dissimilar they are.

  Ward's linkage aims to minimize the increase in the sum of squared distances after merging two clusters. It's a variance minimization approach that often leads to compact and spherical clusters. Ward linkage is the default method in many hierarchical clustering implementations.

## Distance Threshold

