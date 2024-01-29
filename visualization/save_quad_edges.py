import numpy as np
from mesh_helper import read_obj

import argparse
import json

import polyscope as ps
from icecream import ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to obj file.')
    args = parser.parse_args()

    model = args.model
    save_path = model.split('.')[0] + '.json'

    mesh = read_obj(model)

    Q = mesh.faces_quad

    if Q is None:
        exit("Not quad mesh...")

    E = np.vstack([
        np.stack([Q[:, 0], Q[:, 1]], -1),
        np.stack([Q[:, 1], Q[:, 2]], -1),
        np.stack([Q[:, 2], Q[:, 3]], -1),
        np.stack([Q[:, 3], Q[:, 0]], -1)
    ])

    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)

    # ps.init()
    # ps.register_curve_network('quad', mesh.vertices, E)
    # ps.show()

    data = {'E': mesh.vertices[E.reshape(-1,)].reshape(-1,).tolist()}

    with open(save_path, 'w') as f:
        json.dump(data, f)
