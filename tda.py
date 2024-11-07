!pip install umap-learn
!pip install giotto-tda

import os
seed_value= 1
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

import pandas as pd
import re
import matplotlib.pyplot as plt

#import seaborn as sns
import time

import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


import warnings
from sklearn.cluster import KMeans



data = pd.read_csv('/content/status col cell pellet M1 positive(top 5).csv')

class_data= data[['Status']]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label= le.fit_transform(class_data)
label2= pd.DataFrame(label)
label2.columns=["Status"]

data2=data[['HMDB0000641','HMDB0001487','HMDB0010355','HMDB0001178','LMPK12113182','HMDB0002829']]

data2

data3= pd.concat([label2,data2],axis=1)
data3

data4=data[['HMDB0000641','HMDB0001487','HMDB0010355','HMDB0001178','LMPK12113182','HMDB0002829']]
data5= pd.concat([label2,data4],axis=1)
data5

import umap
from gtda.mapper import CubicalCover, make_mapper_pipeline
from sklearn.cluster import DBSCAN
from gtda.mapper.visualization import plot_static_mapper_graph
# Define filter function
filter_func = umap.UMAP(n_neighbors=3,random_state=44)

# Define cover
cover = CubicalCover(kind='uniform', n_intervals=3, overlap_frac=0.5)

# Choose clustering algorithm
clusterer = DBSCAN(eps=10)

# Initialise pipeline
pipe = make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=True,
    n_jobs=-1,
)
plotly_kwargs = {"node_trace": {"marker_colorscale": "jet", "marker_size": 40}}
# Plot Mapper graph
fig = plot_static_mapper_graph(pipe, data5, color_data=data5,plotly_params=plotly_kwargs)
fig.show(config={'scrollZoom': True})

from sklearn.decomposition import PCA
filter_func = PCA(n_components=2,random_state=44)
#umap.UMAP(n_neighbors=5,random_state=44)

# Define cover
cover = CubicalCover(kind='uniform', n_intervals=3, overlap_frac=0.3)

# Choose clustering algorithm
clusterer = DBSCAN(eps=10)

# Initialise pipeline
pipe = make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=True,
    n_jobs=-1,
)
plotly_kwargs = {"node_trace": {"marker_colorscale": "jet"}}
# Plot Mapper graph
fig = plot_static_mapper_graph(pipe, data5, color_data=data5,plotly_params=plotly_kwargs)
fig.show(config={'scrollZoom': True})

import seaborn as sns
import matplotlib.pyplot as plt

corr2 = data5.corr() # We already examined SalePrice correlations
plt.figure(figsize=(6, 5))

heat_plot=sns.heatmap(corr2,
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);

fig = heat_plot.get_figure()
fig.savefig("heat_plot.png", bbox_inches='tight')

fea1 = data[['HMDB0014342']]
fea2 = data[['HMDB0001514']]
fea3 = data[['HMDB0000321']]
fea4 = data[['LMSP01080084']]
fea5 = data[['HMDB0001244']]

fea1= fea1.values

#fea1
#color= ['red' if l == 0 else 'green' for l in label2]
plt.figure(figsize=(3, 2))
#c = ['r' if yy==0 else 'b' for yy in label2]

plt.scatter(label2, fea1, c='r')

label3=label2.values
fea11 = fea1

fig, ax = plt.subplots()
for g in np.unique(label3):
    i = np.where(label3 == g)
    ax.scatter( label3[i],fea11[i], label=g)
ax.legend()
plt.show()
