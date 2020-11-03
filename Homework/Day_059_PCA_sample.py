#!/usr/bin/env python
# coding: utf-8

# # PCA 範例

# (Optional) 若尚未安裝相關套件，執行下一行，然後 restart kernel

# In[ ]:


get_ipython().system('pip3 install --user sklearn')
get_ipython().system('pip3 install --user --upgrade matplotlib')


# 載入套件

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)


# 載入 iris 資料集

# In[2]:


iris = datasets.load_iris()
X = iris.data
y = iris.target


# 設定 模型 估計參數

# In[3]:


centers = [[1, 1], [-1, -1], [1, -1]]
pca = decomposition.PCA(n_components=3)


# 資料建模 並 視覺化 結果

# In[4]:


pca.fit(X)
X = pca.transform(X)

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()


for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[ ]:





