import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from statsmodels.tsa.stattools import coint
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

import time
start = time.perf_counter()

# # -----------------------------------------------------------------------------------------------------------
# data loading
dataset = pd.read_csv('SP500.csv')
tickers_info = pd.read_csv('tickers_info.csv')
dataset.dropna(axis=1,inplace=True)
dataset["Date"] = pd.to_datetime(dataset["Date"])
dataset = dataset.set_index("Date").sort_index()
dataset_phase1 = dataset.loc[dataset.index.year == 2023]

# -----------------------------------------------------------------------------------------------------------
# features loading
features =\
(
    dataset_phase1
    .pct_change()
    .mean()
    * 252
)

features = pd.DataFrame(features)
features.columns = ['Return']

features['Vol'] =\
(
     dataset_phase1
    .pct_change()
    .std()
    * np.sqrt(252)
)

features_arr =\
(
    np
    .asarray([np.asarray(features['Return']),
              np.asarray(features['Vol'])
             ]
            )
    .T
)

scaler = StandardScaler().fit(features_arr)
X =\
(
    pd
    .DataFrame(scaler.fit_transform(features_arr),
               columns = features.columns,
               index = features.index)
)


# -----------------------------------------------------------------------------------------------------------
# clustering
ap = AffinityPropagation()
ap.fit(X)
clust_labels = ap.predict(X)
clustered_series = pd.Series(index = X.index, data = ap.fit_predict(X).flatten()
                            )
counts = clustered_series.value_counts()
ticker_count = counts[counts>1]
counts.sort_index(inplace=True)

print ("Clusters formed: %d" % len(ticker_count)
      )
print ("Pairs to evaluate: %d" % (ticker_count * (ticker_count - 1)
                                 ).sum()
      )

# -----------------------------------------------------------------------------------------------------------
# Clustering visualization
cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_
no_clusters = len(cluster_centers_indices)
X_temp = np.asarray(X)
plt.close("all")

fig = plt.figure(figsize=(10, 5)
                 )
colors = cycle(
    "bgrcmykbgrcmykbgrcmykbgrcmyk")

for k, col in zip(range(no_clusters), colors):

    class_members = labels == k
    if class_members.sum() > 1:
        cluster_center = X_temp[cluster_centers_indices[k]]
        plt.plot(X_temp[class_members, 0], X_temp[class_members, 1], col + ".")
        plt.plot(cluster_center[0], cluster_center[1],
                 "o",
                 markerfacecolor=col,
                 markeredgecolor="k",
                 markersize=14)

        for x in X_temp[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.show()

# -----------------------------------------------------------------------------------------------------------
def find_cointegrated_pairs(data, significance=0.05):
    # Get the number of columns in the data (i.e., number of securities)
    n = data.shape[1]

    # Extract the column names (security names) from the data
    keys = data.keys()

    # List to store pairs of securities that are cointegrated with its score
    output = {}

    # Double loop to go through each combination of securities
    for i in range(n):
        for j in range(i + 1, n):

            # Extract the time series data for the two securities in consideration
            S1 = data[keys[i]]
            S2 = data[keys[j]]

            # Perform the cointegration test between the two securities
            result = coint(S1, S2)

            # Extract the score (test statistic) and p-value from the result
            score = abs(result[0])
            pvalue = result[1]

            if pvalue < significance:
                pair = (keys[i], keys[j])
                output[pair] = score

    return output


# -----------------------------------------------------------------------------------------------------------
cluster_dict = {}

for i, which_clust in enumerate(ticker_count.index):
    tickers = clustered_series[clustered_series == which_clust].index
    output = find_cointegrated_pairs(dataset_phase1[tickers])
    cluster_dict[which_clust]= output

pairs = []
scores = []
for clust in cluster_dict.keys():
    pairs.extend(cluster_dict[clust].keys())
    scores.extend(cluster_dict[clust].values())

print ("Pairs cointegrated: %d" % len(pairs))

# -----------------------------------------------------------------------------------------------------------
same_sector = []
for pair in pairs:
    sector1 = tickers_info.loc[tickers_info['Symbol'] == pair[0], 'GICS Sector'].values[0]
    sector2 = tickers_info.loc[tickers_info['Symbol'] == pair[1], 'GICS Sector'].values[0]
    same_sector.append(sector1 == sector2)

print ("Pairs in same sector: %d" % sum(same_sector))

# -----------------------------------------------------------------------------------------------------------
same_industry = []
for pair in pairs:
    industry1 = tickers_info.loc[tickers_info['Symbol'] == pair[0], 'GICS Sub-Industry'].values[0]
    industry2 = tickers_info.loc[tickers_info['Symbol'] == pair[1], 'GICS Sub-Industry'].values[0]
    same_industry.append(industry1 == industry2)

print ("Pairs in same industry: %d" % sum(same_industry))

# -----------------------------------------------------------------------------------------------------------
# OUTPUT
clustered_pairs = pd.DataFrame({'pair':pairs,
                                'score':scores,
                                'same_sector':same_sector,
                                'same_industry':same_industry
                                }
                               )

clustered_pairs.set_index('pair',inplace=True)

# clustered_pairs.to_csv('clustered_pairs.csv')
## visualize at line 110

end = time.perf_counter()
elapsed_time = end - start
print(f"Total running time: {elapsed_time:.4f} seconds")