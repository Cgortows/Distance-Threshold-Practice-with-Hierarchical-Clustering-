# Distance Threshold Practice with Hierarchical Clustering 

While I was looking into learning more about Customer Segmentation. I discovered that the random datasets I could find on the internet were not that great for visualizng clusters the way I wanted to. I decided to try working with synthetic data and it drove me to a better understanding of clustering, especially Hierarchical Clustersing. In order to visualize the datasets I will be using principal component analysis or PCA.

## Synthetic Datasets
We will start with the default perameters of the make_classification function from sklearn.datasets. The resulting dataset is not too impressive so we will have to make some adjustments. 
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

Since most datasets will have far more than 100 samples, we should increse the n_samples perameter from 100 to 1000 to get a more realistic dataset.

![1000_n_samples](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/1000_samples.png)

With the larger dataset, the n_clusters_per_class and n_classes perameters become more apparent. The graph shows four or five relativly distinct clusters. This is where it comes down to how you choose to cluster your data. With the set perameters, n_clusters_per_class = 2 and n_classes = 2, we should have four clusters. But as we can see it could be four or five. This fuzz or uncertainty of the graph is due to the redundant freatures which we will touch on later.
To better view the clusters lets try using the distance threshold perameter in the fcluster function. 

First lets add method='ward' to the linkage function we were using. This perameter determines how the linkage distances between clusters are calculated during the process of merging clusters. Different linkage methods lead to different structures of hierarchical clusters. In the context of hierarchical clustering, a "linkage distance" is a measure of dissimilarity between two clusters. When combining clusters into larger ones, the linkage distance between them is used to determine how similar or dissimilar they are.

Ward's linkage aims to minimize the increase in the sum of squared distances after merging two clusters. It's a variance minimization approach that often leads to compact and spherical clusters. Ward linkage is the default method in many hierarchical clustering implementations.

## Distance Threshold

The distance threshold is a value that you specify to control the granularity of clustering. It defines a threshold for the linkage distances between clusters, below which clusters will be merged together. Any two clusters with a linkage distance lower than the specified threshold are merged into a single cluster.
We have the distance threshold set at 100 currently. This high of a threshold with only 1000 points doesn't give us any clusters. To get to the specific amount of clusters we want, which is four or five, we can take a look at the dendogram to help us figure out where our distance threshold should lie. 
A dendrogram is used to visualize the arrangement of data points or clusters in a hierarchical structure. It is commonly used to show the process of how clusters are formed and merged as part of the hierarchical clustering algorithm. 

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

With a color_threshold set to one-hundred on the dendogram, we have to eyeball the clusters ourselves.

The y-axis will give us the linkage distance between clusters. This will help us determine the correct distance threshold based on how many clusters we want. The dendogram doesn't decide how many clusters there should be. It only lays them out so we can better decide for oursleves.
Looking at the dendogram we can see that five clusters makes a lot more sense than four. To find the distance threshold we want, we can draw a horizontal line through the graph where it will hit the vertical lines coming up from all the clusters we want isolated. We can see that the threshold is around 20.
Changing the color_threshold on the dendogram will better show us the clusters.

![Dendogram](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/Dendogram_2.png)

Now that we have a distance threshold we can apply it to our scatter plot.

![Distance Threshold of 20](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/ds_20.png)

We chose five clusters because it _looks_ like it fits the data best. This might not be what we want in real world data. Two of the five clusters could share an attribute that is more imporant than what is splitting them apart. Or maybe each of the five clusters has an underlying attribute that can show itself as the outliers of the clusters. If we raise the distance threshold then we widden the area for each cluster.

![Distance Threshold of 35](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/ds_35.png)

Lowering the distance threshold is when we start to get more defined clusters. 

![Distance Threshold of 12](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/ds_12.png)

## Changing up the perameters

With synthetic data sets we can work on a number of different types of clustering problems. By changing the n_clusters_per_class and n_classes perameters we can get a wide range of clusters
When changing those perameters be aware that their product must be less than or equal to two times the n_informative perameter or else the function wont run.
The sum of the n_informative and the n_redundant perameters must be less than the n_features perameter.
For these examples the n_informative and n_redundant perameters will be set to five.

![Two Clusters Four CLasses](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/2_clusters_per_4_classes.png)

This set has two clusters per four classes.

![Three Clusters Four Classes](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/3_clusters_per_4_classes.png)

This set has three clusters per four classes.
Changing the redundant versus the informative features also has an affect on the dataset. These examples will have two clusters per two classes.

![Two Informative Five Redundant](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/2_informative_5_redudnat.png)

This set has two informative features and five redundant features.

![Two Informative Eight Redundant](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/2_informative_8_redundant.png)

This set has two informative features and eight redundant featres.

![Five Informative Five Redundant](https://github.com/Cgortows/Distance-Threshold-Practice-with-Hierarchical-Clustering-/blob/main/Images/5_informative_5_redundant.png)

This set has five informative features and five redundant features.
