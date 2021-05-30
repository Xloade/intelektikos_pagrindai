# %%
from ast import Str
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.mixture import GaussianMixture

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
cluster_num = 3
gm = GaussianMixture(n_components=cluster_num, random_state=0).fit(df)
clusters = gm.predict(df);
# %% scatter plot
fig = plt.figure()
ax = plt.axes(projection ='3d')
# colors = ['red','green','blue','purple']

# fig = plt.figure(figsize=(8,8))
ax.scatter(df[:,0], df[:,1], df[:,2], c=clusters)
ax.set_title("Klusterių grafikas, kai yra " +  str(cluster_num) +" klusteriai")
ax.set_xlabel("backers")
ax.set_ylabel("usd_pledged_real")
ax.set_zlabel("usd_goal_real")
plt.show()

# %%
