from flask import Flask, request, jsonify
import os
import numpy as np
import threading
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import shutil

app = Flask(__name__)

# Directory to save uploaded feature vectors
UPLOAD_FOLDER = Path('./uploaded_feature_vectors')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@app.route('/upload-feature-vectors', methods=['POST'])
def upload_feature_vectors():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to the UPLOAD_FOLDER
    file_path = UPLOAD_FOLDER / file.filename
    file.save(file_path)

    return jsonify({"message": f"File {file.filename} uploaded successfully"}), 200

def load_feature_vectors():
    filePath = "./uploaded_feature_vectors/file"
    loaded_feature_vectors = np.loadtxt(filePath, delimiter=',')

    new_csv_path = "./feature_vectors.csv"
    np.savetxt(new_csv_path, loaded_feature_vectors, delimiter=',', fmt='%.8f')
    print(f"Feature vectors saved to {new_csv_path}")

    df = pd.read_csv(new_csv_path, header=None)
    print(df.head())

    first_vector = df.iloc[[0]].values  # Access the first row as a 2D array
    print(first_vector)

if __name__ == '__main__':
    # load_file_thread = threading.Thread(target=load_feature_vectors)
    # load_file_thread.daemon = True
    # load_file_thread.start()

    df = pd.read_csv("./uploaded_feature_vectors/feature_vectors.csv", header=None)
    data = df.values
    print("Feature Vectors:",data)
    print("Feature Vectors Shape:", data.shape)

    n_clusters=4
    kmeans_model = KMeans(n_clusters)
    kmeans_clus = kmeans_model.fit(data)
    predict_clus = kmeans_model.fit_predict(data)
    print("Predict Shape:", predict_clus.shape)
    print("Predict Cluster Array [0]:", predict_clus[0])
    print("KMeans Cluster: ", kmeans_clus)

    # Count the number of data points in each cluster
    unique, counts = np.unique(predict_clus, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    print("Cluster Distribution:", cluster_distribution)

    # Visualize the clusters
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=predict_clus, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('KMeans Clustering Results')
    plt.colorbar(label='Cluster')
    plt.show()

    app.run(host="0.0.0.0", debug=True, port=5001)  # You can change the port if needed
