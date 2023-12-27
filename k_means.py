import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("Pengunjung_mall.csv")
features = data[['Usia', 'Pendapatan (juta Rp)']]
k = int(input("Masukan Jumlah Cluster : "))
kmeans = KMeans(n_clusters=k, n_init=10)  
data['Cluster_KMeans'] = kmeans.fit_predict(features)

plt.scatter(data['Usia'], data['Pendapatan (juta Rp)'], c=data['Cluster_KMeans'], cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='black', label='Centroids')
plt.xlabel('Usia')
plt.ylabel('Pendapatan (juta Rp)')
plt.title(f'Cluster_KMeans Clustering')
plt.legend()
plt.show()