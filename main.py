import subprocess
import tempfile
import argparse
from find_pairs import *
import cv2
from RANSAC import RANSAC, TransformationType
import random


def load_image_to_tuple(image):
    with tempfile.NamedTemporaryFile(suffix='.png') as t:
        # Pass image through imagemagic
        subprocess.call(['convert', '-flatten', image, t.name])
        # Pass image through extract features
        subprocess.call(['extract_features/extract_features_64bit.ln', '-haraff', '-sift', '-i', t.name, '-DE'])
        with open("{}.haraff.sift".format(t.name)) as f:
            data = f.read()
            lines = [l.strip() for l in data.splitlines()]
            line_tuples = list(map(lambda x: tuple(x.split(' ')), lines))
            return line_tuples


def main(image1, image2):
    p1 = load_image_to_tuple(image1)
    p2 = load_image_to_tuple(image2)

    npp1 = np.asarray(p1[2:], dtype=float)
    npp2 = np.asarray(p2[2:], dtype=float)

    knna, knnb, atoa, btob, distances = k_neirest_neighbours(npp1, npp2, k=200)
    to_paint = points_cohesion(knna, knnb, atoa, btob, distances)
    selected_pairsa = npp1[to_paint]
    selected_knna = knna[to_paint]
    paint(selected_pairsa, npp2, selected_knna, image1, image2)


    # RANSAC
    transformation_tuple_a = RANSAC(selected_pairsa, npp2, selected_knna)
    paint(selected_pairsa, npp2, selected_knna, image1, image2, transformation_tuple_a, TransformationType.AFFINE)

    transformation_tuple_b = RANSAC(selected_pairsa, npp2, selected_knna, transformation=TransformationType.PERSPECTIVE)
    paint(selected_pairsa, npp2, selected_knna, image1, image2, transformation_tuple_b, TransformationType.PERSPECTIVE)


def paint(pointsa, pointsb, knn_array1, image1, image2, transform_tuple=None, transformation=TransformationType.AFFINE):
    iimagea = cv2.imread(image1)
    iimageb = cv2.imread(image2)

    if transform_tuple is not None:
        trans_type, numberOfPoints = (cv2.warpAffine, 3) if transformation == TransformationType.AFFINE else (cv2.warpPerspective, 4)
        padded_points = np.pad(pointsa[:, :2], ((0, 0), (0, 1)), mode='constant', constant_values=1)
        pointsa = np.dot(padded_points, transform_tuple[0])[:, :2]
        iimagea = trans_type(iimagea, transform_tuple[1], (iimagea.shape[1], iimagea.shape[0]))

    new_image = np.concatenate((iimagea, iimageb), axis=0)

    for i in range(pointsa.shape[0]):
            a_row = pointsa[i]
            b_row = pointsb[knn_array1[i]].ravel()
            red = random.randrange(0, 255)
            blue = random.randrange(0, 255)
            green = random.randrange(0, 255)
            cv2.line(new_image, (int(a_row[0]), int(a_row[1])), (int(b_row[0]), int(iimagea.shape[0] + b_row[1])), (red, green, blue), 1)

    cv2.imshow('image', new_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_one', type=str)
    parser.add_argument('image_two', type=str)

    args = parser.parse_args()
    image_one = args.image_one
    image_two = args.image_two
    main(image_one, image_two)
