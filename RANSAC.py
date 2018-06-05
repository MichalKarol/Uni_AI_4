import numpy as np
import cv2
import scipy
import sys



class TransformationType:
    AFFINE = 1,
    PERSPECTIVE = 2,

def RANSAC(selected_pairsa, pairsb, selected_knna, transformation=TransformationType.AFFINE, max_iter=1000):
    iterations = 0
    bestfit = None
    besterror = sys.maxsize

    def affine(x, u):
        temp = np.dot(np.linalg.inv(np.array([
            (x[0][0], x[0][1], 1, 0, 0, 0),
            (x[1][0], x[1][1], 1, 0, 0, 0),
            (x[2][0], x[2][1], 1, 0, 0, 0),
            (0, 0, 0, x[0][0], x[0][1], 1),
            (0, 0, 0, x[1][0], x[1][1], 1),
            (0, 0, 0, x[2][0], x[2][1], 1),
        ])), np.array([
            (u[0][0]),
            (u[1][0]),
            (u[2][0]),
            (u[0][1]),
            (u[1][1]),
            (u[2][1]),
        ]))
        return np.array([
            (temp[0], temp[1], temp[2]),
            (temp[3], temp[4], temp[5]),
            (0, 0, 1),
        ])

    def perspective(x, u):
        temp = np.dot(np.linalg.inv(np.array([
            (x[0][0], x[0][1], 1, 0, 0, 0, -u[0][0]*x[0][0], -u[0][0]*x[0][1]),
            (x[1][0], x[1][1], 1, 0, 0, 0, -u[1][0]*x[1][0], -u[1][0]*x[1][1]),
            (x[2][0], x[2][1], 1, 0, 0, 0, -u[2][0]*x[2][0], -u[2][0]*x[2][1]),
            (x[3][0], x[3][1], 1, 0, 0, 0, -u[3][0]*x[3][0], -u[3][0]*x[3][1]),
            (0, 0, 0, x[0][0], x[0][1], 1, -u[0][1]*x[0][0], -u[0][1]*x[0][1]),
            (0, 0, 0, x[1][0], x[1][1], 1, -u[1][1]*x[1][0], -u[1][1]*x[1][1]),
            (0, 0, 0, x[2][0], x[2][1], 1, -u[2][1]*x[2][0], -u[2][1]*x[2][1]),
            (0, 0, 0, x[3][0], x[3][1], 1, -u[3][1]*x[3][0], -u[3][1]*x[3][1]),
        ])), np.array([
            (u[0][0]),
            (u[1][0]),
            (u[2][0]),
            (u[3][0]),
            (u[0][1]),
            (u[1][1]),
            (u[2][1]),
            (u[3][1]),
        ]))
        return np.array([
            (temp[0], temp[1], temp[2]),
            (temp[3], temp[4], temp[5]),
            (temp[6], temp[7], 1),
        ])

    modelFunction, numberOfPoints, generation = (affine, 3, cv2.getAffineTransform) if transformation == TransformationType.AFFINE else (perspective, 4, cv2.getPerspectiveTransform)

    while iterations < max_iter:

        idx = np.random.choice(selected_pairsa.shape[0], numberOfPoints)

        points_a = selected_pairsa[idx][:, :2]
        points_b = pairsb[selected_knna[idx]][:, :2]

        try:
            model = modelFunction(points_a, points_b).T
        except:
            continue

        padded_points = np.pad(selected_pairsa[:, :2], ((0, 0), (0, 1)), mode='constant', constant_values=1)
        transformed_points = np.dot(padded_points, model)[:, :2]

        error = np.trace(scipy.spatial.distance.cdist(transformed_points, pairsb[selected_knna][:, :2]))

        if error < besterror:
            bestfit = model, points_a, points_b
            besterror = error

        iterations += 1

    return bestfit[0], generation(np.float32(bestfit[1]), np.float32(bestfit[2]))