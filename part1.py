import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Authors:
#Juan Alejandro Ramon Ortega (s6470173)
#Erika Saiz (s5880734)
#Dániel Dávid (s6415997)

wine = load_wine()

dimensions = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_classes = pd.Series(wine.target, name="class")

print(dimensions.shape)               #(samples, features)
print(wine.target_names)              #the class names
print(wine_classes.value_counts())    #samples per class
print(dimensions.head())

#standardization
scaler = StandardScaler()
wine.data= scaler.fit_transform(wine.data)


def do_pca(data, components:int = 2, is_plot:bool = False, target = None, exp_variance:bool = False):
    pca_model = PCA(n_components=components, random_state= 100)
    pca_data= pca_model.fit(data)
    reduced_data= pca_data.transform(data)
    if is_plot:
        sns.scatterplot(x = reduced_data[:,0], y = reduced_data[:,1], hue = target, palette="icefire")
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.title("Two component PCA of Wine samples")
        plt.legend(title = "Class")
        plt.show()

    if exp_variance:
        return pca_data.explained_variance_ratio_
    return reduced_data

def explained_variance_sum(list_var, is_verbose:bool = False, is_plot:bool = False) -> list[float]:
    cumulative_sums = []
    sum = 0
    for i in range(len(list_var)):
        sum += list_var[i]
        cumulative_sums.append(sum)
        if is_verbose:
            print(f"Variance until explained up to and including {i}th component: {sum}")

    if is_plot:
        sns.lineplot(x = [i for i in range(1, 14)], y = cumulative_sums)
        plt.xlabel("Number of components (n)")
        plt.ylabel("Cumulative explained variance (%)")
        plt.title("Cumulative explained variance by number of principal components")
        plt.show()
    return cumulative_sums
