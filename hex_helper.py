import numpy as np
import argparse
from dataclasses import dataclass

import polyscope as ps
from icecream import ic


@dataclass
class HexMesh:
    vertices: np.ndarray
    tets: np.ndarray
    uvws: np.ndarray
    wall_facets: np.ndarray


def read_hexex(path) -> HexMesh:

    with open(path, 'r', encoding='utf-8') as f:
        file_lines = f.readlines()
        base = 0

        NV = int(file_lines[base])
        base += 1

        vertices = np.empty((NV, 3))
        for i in range(NV):
            line = file_lines[base + i]
            data = line.split()

            assert len(data) == 3

            vertices[i] = np.float64(data)

        base += NV
        NT = int(file_lines[base])
        base += 1

        tets = np.empty((NT, 4), dtype=np.int64)
        uvws = np.empty((NT, 4, 3))

        for i in range(NT):
            line = file_lines[base + i]
            data = line.split()

            assert len(data) == 16

            # TODO: Check determinant
            tets[i] = np.int64(data[:4])
            uvws[i] = np.float64(data[4:]).reshape(4, 3)

        base += NT

        if base < len(file_lines):
            NW = int(file_lines[base])
            base += 1

            wall_facets = np.empty((NW, 3), dtype=np.int64)
            # wall_dists = np.empty((NW,))

            for i in range(NW):
                line = file_lines[base + i]
                data = line.split()

                assert len(data) == 4

                wall_facets[i] = np.int64(data[:3])
                # wall_dists[i] = np.float64(data[3:])
        else:
            wall_facets = np.empty((0, 3), dtype=np.int64)

        return HexMesh(vertices, tets, uvws, wall_facets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'hex_path',
        type=str,
        help=
        'Path to tetrahedral mesh with seamless hexahedral parameterization (optionally with walls).'
    )
    args = parser.parse_args()

    hex_mesh = read_hexex(args.hex_path)

    ps.init()
    ps.register_volume_mesh('tet', hex_mesh.vertices, hex_mesh.tets)
    if len(hex_mesh.wall_facets) > 0:
        ps.register_surface_mesh('wall', hex_mesh.vertices,
                                 hex_mesh.wall_facets)
    ps.show()
