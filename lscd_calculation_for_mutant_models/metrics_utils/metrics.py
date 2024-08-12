import numpy as np


def equivalence_partitioning(nsi_per_class, num_classes, ns=None):
    epi_per_class = []
    nc = num_classes - 1
    ns = sum(nsi_per_class)
    for i in range(num_classes):
        epi = (nsi_per_class[i] * nc) / np.maximum(ns, np.finfo(np.float32).eps)
        epi_per_class.append(epi)

    return epi_per_class


def centroid_positioning(samples_per_class, centroids, r):
    cpi_per_class = []
    for i in range(len(samples_per_class)):
        cpi = cpi_for_class(samples_per_class[i], centroids[i], r)
        cpi_per_class.append(cpi)
    return cpi_per_class


def cpi_for_class(samples, centroid, r):
    cpi_non_normalized = 0
    nsi = len(samples)
    for i in range(nsi):
        # cpi_non_normalized += cent(samples[i], centroid, r)
        cpi_non_normalized += cent_general(samples[i], centroid, r)
    cpi = cpi_non_normalized / nsi
    return cpi


def cent(sample, centroid, r):
    if euclidean_dist(sample, centroid) >= r:
        return 1
    else:
        return 0


def cent_general(sample, centroid, r):
    if euclidean_dist_general(sample, centroid) <= r:
        return 1
    else:
        return 0


def bci_for_class(confidences, theta1, theta2):
    bci_non_normalized = 0
    nsi = len(confidences)
    for i in range(nsi):
        bci_non_normalized += bound(confidences[i], theta1, theta2)
    bci = bci_non_normalized / len(confidences)
    return bci


def bound(confidence, theta1, theta2):
    if confidence >= theta1 and confidence <= theta2:
        return 1
    else:
        return 0


def euclidean_dist(a, b):
    ab = a - b
    dist = np.linalg.norm(ab, ord=2)
    return dist


def euclidean_dist_general(a, b):
    max_length = max(len(a), len(b))
    a_masked = np.ma.empty(max_length)
    a_masked.mask = True
    a_masked[0 : len(a)] = a

    b_masked = np.ma.empty(max_length)
    b_masked.mask = True
    b_masked[0 : len(b)] = b

    # ab = a-b
    ab = a_masked - b_masked
    dist = np.linalg.norm(ab.data)
    return dist
