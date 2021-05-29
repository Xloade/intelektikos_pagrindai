from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score


def prepareData():
    df = np.genfromtxt("ks-project_short.csv", delimiter=",", skip_header=1)
    rng = np.random.default_rng(12345)
    rints = rng.integers(low=0, high=df.shape[0], size=1000)
    df = df[rints]
    # ,0category,1main_category,2state,3backers,4country,5usd_pledged_real,6usd_goal_real,7project_time
    df = np.delete(df, [0, 1, 2, 4, 6], 1)
    df = df[np.all([df[:, 0] < 400, df[:, 1] < 50000, df[:, 2] < 50000], axis=0)]
    return df


def calculateModel(df, n_cluster):
    model = KMeans(n_clusters=n_cluster, random_state=0)
    cluster = model.fit_predict(df)

    # score = model.score(df)
    silhouette_avg = silhouette_score(df, cluster)
    silhoutes.append(silhouette_avg)
    inertias.append(model.inertia_)
    print("For n_clusters =", n_cluster,
          "The average silhouette_score is :", silhouette_avg)
    print("For n_clusters =", n_cluster,
          "The inertia is :", model.inertia_)
    return cluster


def draw():
    # %% scatter plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(df[:, 0], df[:, 1], df[:, 2], c=cluster)
    plt.show()


def drawSilhoutes(values):
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(range(2, len(values)+2), values)
    plt.show()


df = prepareData()
silhoutes = []
inertias = []
for i in range(2, 11):
    cluster = calculateModel(df, i)
    draw()

drawSilhoutes(silhoutes)
drawSilhoutes(inertias)

