import numpy as np
from PyXAB.algos.HOO import T_HOO                               # the algorithm
from PyXAB.partition.BinaryPartition import BinaryPartition     # the partition

import matplotlib.pyplot as plt
import seaborn as sns

T = 5000                                            # the number of rounds is 1000
domain = [[0, 1]]                                   # the domain is [[0, 1]]
partition = BinaryPartition                         # the partition chosen is BinaryPartition
algo = T_HOO(rounds=T, domain=domain, partition=partition)    # the algorithm is T_HOO

cumulative_regret = 0
cumulative_regret_list = []

# target_0 = lambda x : x * (1-x) * (4 - np.sqrt(np.abs(np.sin(60 * x))))
# target = lambda x : target_0(x) + target_0(x-0.01)
target = lambda x : -(x-0.)**2 + 1

var = 0.01  # noise variance

points = []
for t in range(1, T+1):
    point = algo.pull(t)
    reward = target(point[0]) +  0.1 * np.random.rand() #np.random.uniform(-var,var)     # uniform noise
    algo.receive_reward(t, reward)
    points.append(point[0])

median = np.median(points)
mean   = np.mean(points)

nodes = algo.partition.get_node_list()[-1]
nodes = [node.domain[0] for node in nodes] 
nodes = sorted(nodes, key=lambda node: node[0])
node_domain_min, node_domain_max = nodes[0][0], nodes[-1][1]
nodes_right = [node[1] for node in nodes]
counts = np.zeros(len(nodes), dtype=int)
indices = np.searchsorted(nodes_right, points, side="left")
for index, point in zip(indices, points):
    if point < node_domain_min or point > node_domain_max: 
        continue
    counts[index] += 1
optimal_node = nodes[np.argmax(counts)]
optimal_point = np.mean(optimal_node)

fig = plt.figure()
plt.subplot(1,3,1)
ax= plt.gca()
x = np.linspace(domain[0][0], domain[0][1], 1000)
z = target(x) 

ax.plot(x, z+var, '-r', alpha = 0.1)
ax.plot(x, z, alpha=0.9)
ax.plot(x, z-var, '-r', alpha = 0.1)
plt.axvline(median, color = 'red')
plt.axvline(mean, color = 'green')
plt.axvline(optimal_point, color = 'blue')

plt.subplot(1,3,2)
plt.plot(range(T), points, 'o', markersize = 1, label = 'point history')
plt.axhline(median, color='red')
plt.axhline(mean, color='green')
plt.axhline(optimal_point, color = 'blue')

plt.subplot(1,3,3)
#sns.kdeplot(points)
plt.hist(points, bins = 'auto')
plt.axvline(median, color = 'red')
plt.axvline(mean, color = 'green')
# plt.axvline(0., color = 'magenta')
plt.axvline(optimal_point, color = 'blue')

plt.legend()
plt.show()
