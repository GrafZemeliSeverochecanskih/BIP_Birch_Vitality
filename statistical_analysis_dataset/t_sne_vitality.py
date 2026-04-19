import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

path = "C:/Users/Tkeli/Documents/BIP_project/Chargement_donnees/brezy"
csv_path = "C:/Users/Tkeli/Documents/BIP_project/Chargement_donnees/birch_trees_bratislava.csv"

df = pd.read_csv(csv_path, sep=";")
df["vitality"] = df["vitality"].astype(str).str.replace(",", ".")
vitality_dict = df.set_index("ID")["vitality"].astype(float).to_dict()


X = []
vitality_colors = []

## data loading
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)

    if not os.path.isdir(folder_path):
        continue

    vitality = vitality_dict.get(int(folder), None) 

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg")):
            img = Image.open(os.path.join(folder_path, file)).convert("RGB")
            img = img.resize((64, 64))

            X.append(np.array(img).flatten())
            vitality_colors.append(vitality)  

X = np.array(X)

## t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca"
)

X_tsne = tsne.fit_transform(X)

## plotting
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=vitality_colors, cmap="RdYlGn", s=5)
plt.colorbar(scatter, label="Vitality level")

plt.title("t-SNE Projection of Images (RGB)", fontsize=18)
plt.xlabel("Dimension 1", fontsize=15)
plt.ylabel("Dimension 2", fontsize=15)

plt.show()