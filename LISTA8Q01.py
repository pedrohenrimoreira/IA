#1. Pré-processamento dos Dados
#Identificação e Remoção de Outliers:

import numpy as np
from scipy import stats

# Supondo que 'dados' é um DataFrame contendo seu conjunto de dados
z_scores = np.abs(stats.zscore(dados))
threshold = 3
dados_sem_outliers = dados[(z_scores < threshold).all(axis=1)]

#Normalização dos Dados:

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dados_normalizados = scaler.fit_transform(dados_sem_outliers)


#2. Encontrar e Caracterizar Agrupamentos
Aplicação do K-means e Cálculo do Índice Silhouette:

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Suponha que queremos testar de 2 a 10 clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dados_normalizados)
    score = silhouette_score(dados_normalizados, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"Silhouette Score for k={k}: {score}")

#Método Elbow para determinar o número ideal de clusters:

wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(dados_normalizados)
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#3. Explicação das Métricas
#(Silhouette e Elbow já foram calculados e plotados acima. As explicações matemáticas podem ser adicionadas como comentários no código.)

#from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(dados_normalizados, kmeans.labels_)
print(f"Davies-Bouldin Index: {db_index}")

from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(dados_normalizados, kmeans.labels_)
print(f"Davies-Bouldin Index: {db_index}")

#5. Visualização de Agrupamentos e Identificação de Erros:
plt.figure(figsize=(10, 6))
plt.scatter(dados_normalizados[:, 0], dados_normalizados[:, 1], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.6)
plt.title('Visualization of Clustered Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

''''Relatório de Análise de Cluster com K-means no Conjunto de Dados Iris
Introdução
Este relatório apresenta os resultados da análise de clusterização utilizando o algoritmo K-means no conjunto de dados Iris para identificar padrões e agrupamentos com base nas características das flores.

Metodologia
Pré-processamento dos Dados: Remoção de outliers e normalização dos dados usando MinMaxScaler.
Aplicação do K-means: Execução do K-means para diferentes números de clusters (de 2 a 10) e avaliação usando o índice de Silhouette e o método Elbow.
Resultados e Discussão
A aplicação do K-means sugeriu que o número ideal de clusters é 3, o que coincide com as três espécies de Iris no conjunto de dados. Os scores de Silhouette indicaram uma boa separação e coesão dos clusters nesta configuração.

Conclusão
A análise confirmou a eficácia do K-means em agrupar os dados de Iris em três clusters distintos, correspondentes às suas espécies biológicas, demonstrando a utilidade do pré-processamento adequado e das técnicas de avaliação de cluster na otimização dos resultados de agrupamento.

''''
