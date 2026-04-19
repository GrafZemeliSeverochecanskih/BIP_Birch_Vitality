import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/Tkeli/Documents/BIP_project/Chargement_donnees/brezy"

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)

    if not os.path.isdir(folder_path):
        continue

    hist_total = np.zeros(256)
    count = 0

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg")):
            img_path = os.path.join(folder_path, file)

            img = Image.open(img_path).convert("L") #gray level conversion
            gray = np.array(img)

            hist, bin = np.histogram(gray, bins=256, range=(0, 256))
            hist_total += hist
            count += 1

    if count > 0:
        hist_avg = hist_total / count
        hist_avg = hist_avg / hist_avg.sum() 

        print(hist_avg)

        plt.plot(hist_avg, label=folder)
       

plt.title("Pixel Intensity Distribution per Tree", fontsize=17)
plt.xlabel("Intensity (0 = black, 255 = white)",fontsize=15)
plt.ylabel("Proportion of Pixels",fontsize=15)

plt.show()
    