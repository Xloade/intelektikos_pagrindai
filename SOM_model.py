# %%
from sklearn_som.som import SOM
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# %% prepare data
df = np.genfromtxt("ks-project_short.csv", delimiter=",", skip_header=1)
rng = np.random.default_rng(12345)
rints = rng.integers(low=0, high=df.shape[0], size=1000)
df = df[rints]
# ,0category,1main_category,2state,3backers,4country,5usd_pledged_real,6usd_goal_real,7project_time
df = np.delete(df, [0,1,2,4,7], 1)
# %%
df = df[np.all([df[:,0]<400, df[:,1]<50000, df[:,2]<50000], axis=0)]
#%% train model
model = SOM(m=50, n=50, dim=3)
model.fit(df)
cluster = model.fit_predict(df)
# %% scatter plot
fig = plt.figure()
ax = plt.axes(projection ='3d')
# colors = ['red','green','blue','purple']

# fig = plt.figure(figsize=(8,8))
ax.scatter(df[:,0], df[:,1], df[:,2], c=cluster)

# cb = plt.colorbar()
# loc = np.arange(0,max(label),max(label)/float(len(colors)))
# cb.set_ticks(loc)
# cb.set_ticklabels(colors)
#%%
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.scatter(df[:,0], df[:,1], df[:,2], alpha=0.1)
cluster_centers = np.concatenate(model.cluster_centers_, axis=1)
ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], color="r")

plt.show()
# %%
# %%
