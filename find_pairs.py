import numpy as np
import functools as f
# import cupy as cp
import scipy.spatial


def distance(x, pointsb):
    return np.linalg.norm(x.reshape(1, 128) - pointsb, axis=1)


def calc_distance(pointsa, pointsb):
    return np.array([distance(v, pointsb) for i, v in enumerate(pointsa)])


def k_neirest_neighbours(pairsa, pairsb, k=50):
    features_diffrence = scipy.spatial.distance.cdist(pairsa[:, 5:], pairsb[:, 5:], 'euclidean')
    knna = np.argmin(features_diffrence, axis=1)
    knnb = np.argmin(features_diffrence.transpose(), axis=1)

    tree = scipy.spatial.cKDTree(pairsa[:, :2])

    def get_k_closest(row):
        _, ii = tree.query(row, k=k+1)
        return ii[1:]

    atoa = np.apply_along_axis(get_k_closest, arr=pairsa[:, :2], axis=1)
    tree = scipy.spatial.cKDTree(pairsb[:, :2])
    btob = np.apply_along_axis(get_k_closest, arr=pairsb[:, :2], axis=1)

    return knna, knnb, atoa, btob, features_diffrence

def row_cohesion(x, knna, knnb, atoa, btob, distances, index):
    x_o = x

    if knnb[x_o].ravel() != index:
        return False

    # z_o = distances[index, x_o]
    ff = atoa[index]
    fg = btob[x_o]

    diff = len(set(knna[ff].flat).difference(set(fg.flat)))
    return diff <= 0.5 * ff.size


def calc_cohesion(knna, knnb, atoa, btob, distances):
    return np.array([row_cohesion(v, knna, knnb, atoa, btob, distances, i) for i, v in enumerate(knna)])


def points_cohesion(knna, knnb, atoa, btob, distances):
    res = calc_cohesion(knna, knnb, atoa, btob, distances)
    return res