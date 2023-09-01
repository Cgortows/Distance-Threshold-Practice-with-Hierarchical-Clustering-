# Distance Threshold Practice with Hierarchical Clustering 

While I was exploring Customer Segmentation , I realized that the random datasets available on the internet weren't ideal for visualizing clusters the way I intended. Consequently, I chose to experiment with synthetic data, which led me to a deeper comprehension of clustering, particularly Hierarchical Clustering. To effectively visualize the datasets, I will employ Principal Component Analysis (PCA).

## Synthetic Datasets
We'll begin by utilizing the default parameters of the `make_classification` function from the `sklearn.datasets` module. However, the resulting dataset is not very impressive, so we will have to make some adjustments.

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
distance_threshold = 100
clusters = fcluster(Z_pca_complex, distance_threshold, criterion='distance')

# Scatter plot of the hierarchical clustering results with discrete colormap
plt.figure(figsize=(8, 6))
plt.scatter(df_pca_complex['PC1'], df_pca_complex['PC2'], c=clusters, cmap='tab10', s=30, alpha=0.5)
plt.title('Default Settings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
```

![Default settings](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/default_make_classification_settings.png)

Given that the majority of datasets typically consist of more than 100 samples, we will increase the `n_samples` parameter from 100 to 1000.

![1000_n_samples](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/1000_samples.png)

With the increased dataset size, the significance of the `n_clusters_per_class` and `n_classes` parameters becomes more evident. The graph depicts four or five relatively distinct clusters. The decision of how to cluster the data now becomes crucial. With the specified parameters, where `n_clusters_per_class = 2` and `n_classes = 2`, we anticipate having four clusters. However the number could be four or five. This fuzz or uncertainty of the graph is due to the redundant freatures which we will touch on later.

To gain a clearer view of the clusters, let's experiment with using the `distance_threshold` parameter in the `fcluster` function.

First, let's include the `method='ward'` parameter in the `linkage` function we've been using. This parameter defines how the linkage distances between clusters are computed during the merging process. Different linkage methods result in distinct hierarchical cluster structures. In the context of hierarchical clustering, a "linkage distance" quantifies the dissimilarity between two clusters. When combining clusters into larger ones, the linkage distance guides the determination of their similarity or dissimilarity.

Ward's linkage aims to minimize the increase in the sum of squared distances after merging two clusters. It's a strategy focused on minimizing variance and frequently yields compact and spherical clusters. Ward linkage serves as the default method in many hierarchical clustering implementations.

## Distance Threshold

The distance threshold is used to regulate the level of detail in clustering. It establishes a threshold for linkage distances between clusters, below which clusters are combined. When the linkage distance between any two clusters is lower than the specified threshold, those clusters are merged into a single cluster.

Currently, we have set the distance threshold at 100. However, this high threshold, given the dataset size of 1000 points, prevents the formation of distinct clusters. To get to four or five clusters, we can utilize the dendrogram as a guide to determine the appropriate placement of our distance threshold.

A dendrogram serves as a visualization tool to depict the hierarchical arrangement of data points or clusters. It is commonly employed to illustrate the process of cluster formation and merging within the context of the hierarchical clustering algorithm. The dendogram function has a `color_threshold` parameter which you can think of as a distance threshold.

```
# Create a dendrogram
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram (PCA Transformed Complex Data)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(Z_pca_complex, leaf_rotation=90, leaf_font_size=8, color_threshold=100)
plt.show()
```

![Dendogram](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/dendogram.png)

With a `color_threshold` set to one hundred on the dendrogram, we need to visually assess the clusters ourselves.

The y-axis on the dendrogram represents the linkage distance between clusters. This information aids us in determining the appropriate distance threshold based on our desired number of clusters. However, it's important to note that the dendrogram itself doesn't dictate the number of clusters; it merely presents the data so that we can make informed decisions.

Upon examining the dendrogram, we can see that five clusters make more sense than four.

To determine the desired distance threshold, we can draw a horizontal line across the graph where it intersects the vertical lines extending from the clusters we wish to isolate. It appears that the threshold is approximately 20.

For improved clarity in observing the clusters, we can modifying the color_threshold on the dendrogram.

![Dendogram](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/Dendogram_2.png)

Now that we have established a distance threshold, we can implement it within our scatter plot. This threshold will guide the formation and visualization of clusters in our scatter plot based on the hierarchical clustering results.

![Distance Threshold of 20](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/ds_20.png)

We opted for five clusters because it visually seems to align well with the data's structure. However, this choice might not be ideal for real-world data scenarios. It's possible that two out of the five clusters could share a crucial attribute that outweighs the factors causing them to appear distinct. Alternatively, each of the five clusters could possess an intrinsic attribute that manifests as outliers within the clusters.

Raising the distance threshold would result in broader regions for each cluster. This adjustment could lead to clusters capturing more diverse points and potentially revealing underlying attributes that might not have been apparent with a smaller threshold. It's essential to strike a balance between granularity and meaningful interpretation of the clusters, considering the characteristics of your specific data and objectives.

![Distance Threshold of 35](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/ds_35.png)

Lowering the distance threshold would result in narrower regions for each cluster. This adjustment could lead to clusters capturing tighter and more homogenous groupings of points, which may unveil finer and more subtle patterns in the data.

![Distance Threshold of 12](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/ds_12.png)

## Changing up the perameters

With synthetic datasets, we have the flexibility to explore various types of clusters. By adjusting the `n_clusters_per_class` and `n_classes` parameters, we can generate a diverse array of clusters.

When making changes to these parameters, it's important to be mindful that their product must not exceed two times the value of the `n_informative` parameter; otherwise, the function will not execute. This constraint is in place to ensure that the number of possible classes and clusters doesn't surpass the potential informative feature combinations.

Additionally, the sum of the `n_informative` and `n_redundant` parameters should be smaller than the `n_features` parameter.

For the examples presented here, the `n_informative` and `n_redundant` parameters will both be set to five.

![Two Clusters Four CLasses](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/2_clusters_per_4_classes.png)

This set has two clusters per four classes.

![Three Clusters Four Classes](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/3_clusters_per_4_classes.png)

This set has three clusters per four classes.
Manipulating the balance between redundant and informative features also exerts a significant influence on the resulting dataset. By altering the proportion of these features, we can further tailor the characteristics of the dataset to our needs. In the cases presented, the configuration will involve two clusters within each of the two classes.

The interplay between redundant and informative features plays a pivotal role in shaping the data's structure. Redundant features might contribute to increased complexity, while informative features enhance the discriminative power of the dataset. This delicate equilibrium between the two types of features allows us to fine-tune the clustering outcomes to match the inherent properties of the data.

![Two Informative Five Redundant](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/2_informative_5_redudnat.png)

This set has two informative features and five redundant features.

![Two Informative Eight Redundant](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/2_informative_8_redundant.png)

This set has two informative features and eight redundant featres.

![Five Informative Five Redundant](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/5_informative_5_redundant.png)

This set has five informative features and five redundant features.

In conclusion, this exploration into Customer Segmentation using synthetic datasets and hierarchical clustering has provided valuable insights into the intricate world of cluster analysis. By meticulously adjusting parameters, scrutinizing dendrograms, and discerning the interplay between features, we've uncovered how each facet shapes the clustering outcomes. Through the lens of synthetic data, we've not only deepened our understanding of clustering algorithms but also honed the art of data interpretation. This project underscores the importance of striking a balance between algorithmic sophistication and insightful domain knowledge to uncover hidden patterns within complex datasets. As we journey further into the realm of data analysis, the lessons gleaned from this project will undoubtedly serve as a compass, guiding us through the nuanced terrain of clustering and paving the way for more informed decision-making in real-world scenarios
