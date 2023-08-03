import igl
import numpy as np
import polyscope as ps
from icecream import ic

from common import normalize
from extrinsically_smooth_direction_field import per_face_basis

V, F = igl.read_triangle_mesh('data/mesh/sphere.obj')

one_ring = [57, 7, 25, 59, 9, 86]
two_ring = [66, 50, 31, 91, 8, 69, 36, 90, 99, 62, 77, 82, 60, 1, 78, 79, 34]

vis_scalar = np.zeros(len(F))
vis_scalar[one_ring] = 0.5
vis_scalar[two_ring] = 1.0

FN, Tf = per_face_basis(V[F])


def R2(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


# https://stackoverflow.com/questions/14066933/direct-way-of-computing-the-clockwise-angle-between-two-vectors
def oriented_angle(x, y, n):
    dot = np.dot(x, y)
    det = np.dot(n, np.cross(x, y))
    return np.arctan2(det, dot)


def angle_diff(fi, fj):
    # x_i
    e01_i = [F[fi][1], F[fi][0]]
    e02_i = [F[fi][2], F[fi][0]]
    e12_i = [F[fi][2], F[fi][1]]
    e_i = [e01_i, e02_i, e12_i]

    # x_j
    e01_j = [F[fj][1], F[fj][0]]
    e02_j = [F[fj][2], F[fj][0]]
    e12_j = [F[fj][2], F[fj][1]]
    e_j = [e01_j, e02_j, e12_j]

    # https://stackoverflow.com/questions/653509/breaking-out-of-nested-loops
    for i in range(3):
        for j in range(3):
            if e_i[i][0] == e_j[j][1] and e_i[i][1] == e_j[j][0]:
                break
        else:
            continue
        break

    if i == 0 and j == 0:
        R = R2(np.pi)
    elif i == 0:
        x_j = normalize(V[e_j[0][0]] - V[e_j[0][1]])
        edge = normalize(V[e_i[0][0]] - V[e_i[0][1]])
        theta_ji = oriented_angle(x_j, edge, FN[fj])
        R = R2(-theta_ji)
    elif j == 0:
        x_i = normalize(V[e_i[0][0]] - V[e_i[0][1]])
        edge = normalize(V[e_i[i][0]] - V[e_i[i][1]])
        theta_ij = oriented_angle(x_i, -edge, FN[fi])
        R = R2(theta_ij)
    else:
        x_i = normalize(V[e_i[0][0]] - V[e_i[0][1]])
        edge = normalize(V[e_i[i][0]] - V[e_i[i][1]])
        x_j = normalize(V[e_j[0][0]] - V[e_j[0][1]])

        theta_ij = oriented_angle(x_i, edge, FN[fi])
        theta_ji = oriented_angle(x_j, edge, FN[fj])

        R = R2(theta_ij - theta_ji)

    return R


uv = np.array([0.75, 0.55])
tangents = np.zeros_like(F, dtype=np.float64)
tangents[two_ring[0]] = Tf[two_ring[0]].T @ uv

for i in range(len(two_ring) - 1):
    uv = angle_diff(two_ring[i], two_ring[i + 1]) @ uv
    tangents[two_ring[i + 1]] = Tf[two_ring[i + 1]].T @ uv

ps.init()
vis_mesh = ps.register_surface_mesh("sphere", V, F)
vis_mesh.add_scalar_quantity("vis_scalar",
                             vis_scalar,
                             defined_on='faces',
                             enabled=True)
vis_mesh.add_vector_quantity("x", Tf[:, 0], defined_on='faces')
vis_mesh.add_vector_quantity("y", Tf[:, 1], defined_on='faces')
vis_mesh.add_vector_quantity("tangent",
                             tangents,
                             defined_on='faces',
                             enabled=True)
ps.show()
