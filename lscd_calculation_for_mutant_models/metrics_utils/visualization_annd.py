import matplotlib.pyplot as plt
import numpy as np

# Arrays containing ANND values for corresponding index as the number of nearest neighbors.
kmnc = np.genfromtxt('annds/annds_svhn_kmnc.csv', delimiter=',')[0:200]
org_train = np.genfromtxt('annds/annds_svhn_train.csv', delimiter=',')[0:200]
org_test = np.genfromtxt('annds/annds_svhn_test.csv', delimiter=',')[0:200]

plt.plot(np.arange(kmnc.shape[0]), kmnc, color='red', label='kmnc')
plt.plot(np.arange(org_train.shape[0]), org_train, color='cyan', label='train')
plt.plot(np.arange(org_test.shape[0]), org_test, color='blue', label='test')

# Visualize collisions of graphs
for i in range(2, len(kmnc)):
    if (org_train[i] > kmnc[i] and org_train[i-1] <= kmnc[i-1]) or (org_train[i] > org_test[i] and org_train[i-1] <= org_test[i-1]):
        plt.plot(i, org_train[i], marker='x', markersize=5, color="black")
    if (kmnc[i] > org_train[i] and kmnc[i-1] <= org_train[i-1]) or (kmnc[i] > org_test[i] and kmnc[i-1] <= org_test[i-1]):
        plt.plot(i, kmnc[i], marker='x', markersize=5, color="black")
    if (org_test[i] > kmnc[i] and org_test[i-1] <= kmnc[i-1]) or (org_test[i] > org_train[i] and org_test[i-1] <= org_train[i-1]):
        plt.plot(i, org_test[i], marker='x', markersize=5, color="black")
    


plt.title('Comparison of ANNDs: Train vs Test vs Corner Case')
plt.xlabel('Number of nearest neighbours')
plt.ylabel('Average nearest neighbour distance')

plt.legend()
plt.show()
