#messy import since UKGE and the dataset are in a parent subdirectory
import sys
sys.path.append(".")

import torch
import numpy as np
from sklearn.decomposition import PCA
from UKGE.KGDataset import KGDataset
from matplotlib import animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os

#number of pooints for the plot
#about 4 mins to run for 50K points on my machine
#140K points maximum, since that the number of entites in the test set(70k from each graph)
num_points = 50000

# Load the PyTorch model
data_dir = "./data/psl50/OpenEA/"
data_set = "EN_DE_100K_V2"
model_dir = "./results/models/psl50/EN_DE_100K_V2/1/"
model_name = max([f for f in os.listdir(model_dir) if f.endswith(".model")], key=lambda x: float(x[:-6]))
model = torch.load(model_dir+model_name)
model.to("cpu")

train_set = KGDataset(data_dir + data_set + "/1/kg1_list.tsv")
train_set.load_kg(data_dir + data_set + "/1/kg2_list.tsv")
train_set.load_kg(data_dir + data_set + "/1/kg_atts.tsv")
train_set.load_kg(data_dir + data_set + "/1/link_list.tsv")

#Read in a list of index names and graph ids from file, convert the index names to indices thorugh the model.lookup_ents dictionary, and group them based on graph id in an array
points = {}
for line in open(f"{data_dir}{data_set}/ent_ids.tsv"):
    line = line.rstrip('\n').split('\t')
    if line[0] not in train_set.index_ents:
        continue
    if line[1] not in points:
        points[line[1]] = [train_set.index_ents[line[0]]]
    else:
        points[line[1]].append(train_set.index_ents[line[0]])

#extract the embeddings for the points based on graph id
embeddings = {}
for key in points.keys():
    for emb in points[key]:
        emb = torch.tensor([emb], dtype=torch.int)
        if key not in embeddings:
            embeddings[key] = [model.entityEmbed(emb).squeeze().detach().numpy()]
        else:
            embeddings[key].append(model.entityEmbed(emb).squeeze().detach().numpy())

#combine the two sets of embeddings but keep track of the group they belong to
v_len = len(embeddings['1'])
emb_group = list(embeddings['1']) + list(embeddings['2'])

#extract a random subset of the embeddings for PCA
rand_indices = np.random.randint(0, len(emb_group), num_points)
emb_group = [emb_group[i] for i in rand_indices]
assigned_group = [0 if i < v_len else 1 for i in rand_indices]

# Run PCA on each group of embeddings and plot the results
pca = PCA(n_components=3)
result = pca.fit_transform(emb_group)
plt.figure(figsize=(10, 6))

#get each group of points based on the assigned group
group1 = np.array([result[i] for i in range(len(result)) if assigned_group[i] == 0])
group2 = np.array([result[i] for i in range(len(result)) if assigned_group[i] == 1])

#color the two groups of points differently by making negtaive and positive different maps
g_x, g_y, g_z = zip(*np.concatenate([group1,group2],0))
neg_vals = np.sqrt(np.sum(group2**2,-1))
dists = np.hstack((np.sqrt(np.sum(group1**2,-1)), neg_vals - np.max(neg_vals)))
color1 = plt.cm.autumn(np.linspace(0., 1, 128))
color2 = plt.cm.winter(np.linspace(0., 1, 128))
colors = np.vstack((color1, color2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

#plot the data
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(g_x, g_y, g_z, alpha=0.1, cmap=mymap, c=dists, label="1")
#label plot
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('PCA Plot of Embeddings')

def rotate(angle):
    ax.view_init(azim=angle)

rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)

plt.savefig('pca_plot.png', dpi=300)
print()
rot_animation.save('rotation.mp4', dpi=200, writer=animation.FFMpegWriter(fps=15), progress_callback=lambda i, n: print(f"\r{100*i/n:.2f}%", end=""))
