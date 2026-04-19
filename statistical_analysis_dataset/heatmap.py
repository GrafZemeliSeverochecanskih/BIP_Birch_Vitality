import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

##data loading
path = "C:/Users/Tkeli/Documents/BIP_project/Chargement_donnees/birch_trees_bratislava.csv"
df = pd.read_csv(path, sep = ";")  


##average intensity per folder (per tree)
tree_brightness={}
root_folder = "C:/Users/Tkeli/Documents/BIP_project/Chargement_donnees/brezy"

for folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder)

    if not os.path.isdir(folder_path):
        continue

    values = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg")):
            img_path = os.path.join(folder_path, file)

            img = Image.open(img_path).convert("L")
            brightness = np.array(img).mean()
            values.append(brightness)

    if values:
        tree_brightness[folder] = np.mean(values)


df["brightness"] = df["ID"].astype(str).map(tree_brightness)


#division of brightness in 3 categories 
df["brightness_category"] = pd.cut(
    df["brightness"],
    bins=3,
    labels=["dark", "medium", "bright"]
)

##heatmap
heatmap_data = pd.crosstab(df["vitality"], df["brightness_category"]) #count number of image / categorie


## graphique
plt.figure(figsize=(8, 5))
plt.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")


plt.xticks(ticks=[0, 1, 2], labels=["dark", "medium", "bright"])
plt.yticks(ticks=range(len(heatmap_data.index)), labels=heatmap_data.index)
plt.xlabel("Brightness")
plt.ylabel("Vitality level")
plt.title("Heatmap: Vitality vs Brightness")

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        plt.text(j, i, heatmap_data.iloc[i, j],
                 ha="center", va="center", color="black")

plt.colorbar(label="Number of images")
plt.tight_layout()
plt.show()