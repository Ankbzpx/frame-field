import numpy as np
import os
from PIL import Image
import trimesh
from dataclasses import dataclass, field

import polyscope as ps
from icecream import ic


@dataclass
class OBJMesh:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray | None = None
    faces_quad: np.ndarray | None = None
    uvs: np.ndarray | None = None
    face_uvs_idx: np.ndarray | None = None
    face_uvs_idx_quad: np.ndarray | None = None
    materials: list[Image.Image] = field(default_factory=lambda: [])
    vertex_colors: np.ndarray | None = None
    polygon_groups: np.ndarray | None = None
    polygon_groups_quad: np.ndarray | None = None
    extras: list[str] = field(default_factory=lambda: [])


# TODO: Support multiple materials, face normals
# Modified from https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.obj.html#module-kaolin.io.obj for vertex color support
def read_obj(path, warning=False) -> OBJMesh:
    r"""
    Load obj, assume same face size. Support quad mesh, vertex color, polygon_groups
    """
    vertices = []
    faces = []
    uvs = []
    vertex_normals = []
    vertex_colors = []
    face_uvs_idx = []
    polygon_groups = []
    mtl_path = None
    materials = []
    extras = []

    with open(path, 'r', encoding='utf-8') as f:
        file_lines = f.readlines()
        for i in range(len(file_lines)):
            line = file_lines[i]
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == 'v':
                vertices.append(np.float64(data[1:4]))
                if len(data[4:]) != 0:
                    vertex_colors.append(np.float64(data[4:]))
            elif data[0] == 'vt':
                uvs.append(np.float64(data[1:3]))
            elif data[0] == 'vn':
                vertex_normals.append(np.float64(data[1:]))
            elif data[0] == 'f':
                data = [da.split('/') for da in data[1:]]
                faces.append([int(d[0]) for d in data])
                if len(data[1]) > 1 and data[1][1] != '':
                    face_uvs_idx.append([int(d[1]) for d in data])
                else:
                    face_uvs_idx.append([0] * len(data))
            elif data[0] == 'g':
                polygon_groups.append(i)
            elif data[0] == 'mtllib':
                extras.append(line)
                mtl_path = os.path.join(os.path.dirname(path), data[1])
                if os.path.exists(mtl_path):
                    with open(mtl_path) as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith("map_Kd"):
                                texture_file = line.split(' ')[-1]
                                texture_file = texture_file.strip('\n')
                                texture_file_path = os.path.join(
                                    os.path.dirname(path), texture_file)
                                img = Image.open(texture_file_path)
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                materials.append(img)
                else:
                    if warning:
                        print(
                            f"Failed to load material, {data[1]} doesn't exist")

    vertices = np.stack(vertices).reshape(-1, 3)

    if len(uvs) == 0:
        uvs = None
    else:
        uvs = np.stack(uvs).reshape(-1, 2)

    if len(vertex_colors) == 0:
        vertex_colors = None
    else:
        vertex_colors = np.stack(vertex_colors).reshape(-1, 3)

    if len(face_uvs_idx) == 0:
        face_uvs_idx = None
    else:
        face_uvs_idx = np.int64(face_uvs_idx) - 1

    faces = np.int64(faces) - 1

    if len(polygon_groups) != 0:
        polygon_groups = np.array(polygon_groups)
        polygon_groups -= polygon_groups[0]
        polygon_groups -= np.arange(len(polygon_groups))
    else:
        polygon_groups = None

    faces_quad = None
    face_uvs_idx_quad = None
    polygon_groups_quad = None
    face_size = faces.shape[1]
    if face_size == 4:
        faces_quad = faces
        # Quad face splitting matches Libigl and three.js
        faces = np.vstack([
            np.vstack([[face[0], face[1], face[2]],
                       [face[0], face[2], face[3]]]) for face in faces_quad
        ])
        if face_uvs_idx is not None:
            face_uvs_idx_quad = face_uvs_idx
            face_uvs_idx = np.vstack([
                np.vstack([[face_uv_idx[0], face_uv_idx[1], face_uv_idx[2]],
                           [face_uv_idx[0], face_uv_idx[2], face_uv_idx[3]]])
                for face_uv_idx in face_uvs_idx_quad
            ])
        if polygon_groups is not None:
            polygon_groups_quad = polygon_groups
            polygon_groups = 2 * polygon_groups_quad

    if len(vertex_normals) == 0 or len(vertex_normals) != len(vertices):
        if warning:
            print("Obj doesn't contain vertex_normals, compute using trimesh")
        vertex_normals = np.copy(
            trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
                maintain_order=True,
            ).vertex_normals)
        assert not np.isnan(vertex_normals).any()
        assert len(vertex_normals) == len(vertices)
    else:
        vertex_normals = np.stack(vertex_normals).reshape(-1, 3)

    return OBJMesh(vertices, faces, vertex_normals, faces_quad, uvs,
                   face_uvs_idx, face_uvs_idx_quad, materials, vertex_colors,
                   polygon_groups, polygon_groups_quad, extras)


# TODO: Support multiple materials write, face normal
def write_obj(filename, mesh: OBJMesh, face_group_id=None):
    r"""
    Write obj, support quad mesh
    """
    with open(filename, 'w', encoding='utf-8') as obj_file:
        for extra in mesh.extras:
            if extra.split()[0] == 'mtllib':
                obj_file.write(extra)

        if mesh.vertex_colors is None:
            for v in mesh.vertices:
                obj_file.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        else:
            for v, vc in zip(mesh.vertices, mesh.vertex_colors):
                obj_file.write('v %f %f %f %f %f %f\n' %
                               (v[0], v[1], v[2], vc[0], vc[1], vc[2]))

        if mesh.uvs is not None:
            for vt in mesh.uvs:
                obj_file.write('vt %f %f\n' % (vt[0], vt[1]))

        group_counter = 0
        sort_idx = None
        if face_group_id is not None:
            if mesh.faces_quad is None:
                face_idx_all = np.arange(len(face_group_id))
                sort_idx = np.concatenate([
                    face_idx_all[face_group_id == id]
                    for id in np.arange(face_group_id.max() + 1)
                ])
            else:
                face_group_id = face_group_id[::2]
                sort_idx = np.argsort(face_group_id)
            _, count = np.unique(face_group_id[sort_idx],
                                 return_counts=True,
                                 axis=0)
            polygon_groups = np.cumsum(count)
            polygon_groups = [0] + list(polygon_groups[:-1])
        elif mesh.polygon_groups is not None:
            if mesh.faces_quad is None:
                polygon_groups = mesh.polygon_groups
            else:
                polygon_groups = mesh.polygon_groups_quad
        else:
            polygon_groups = None

        if mesh.faces_quad is not None:
            if sort_idx is not None:
                faces_quad = mesh.faces_quad[sort_idx]
                face_uvs_idx_quad = mesh.face_uvs_idx_quad[
                    sort_idx] if mesh.face_uvs_idx_quad is not None else None
            else:
                faces_quad = mesh.faces_quad
                face_uvs_idx_quad = mesh.face_uvs_idx_quad

            for i in range(len(faces_quad)):
                if polygon_groups is not None:
                    if group_counter < len(
                            polygon_groups
                    ) and i == polygon_groups[group_counter]:
                        obj_file.write(f"g {group_counter}\n")
                        group_counter += 1

                f = faces_quad[i]
                if face_uvs_idx_quad is not None:
                    f_uv = face_uvs_idx_quad[i]
                    obj_file.write(
                        'f %d/%d %d/%d %d/%d %d/%d\n' %
                        (f[0] + 1, f_uv[0] + 1, f[1] + 1, f_uv[1] + 1, f[2] + 1,
                         f_uv[2] + 1, f[3] + 1, f_uv[3] + 1))
                else:
                    obj_file.write('f %d %d %d %d\n' %
                                   (f[0] + 1, f[1] + 1, f[2] + 1, f[3] + 1))
        else:
            if sort_idx is not None:
                faces = mesh.faces[sort_idx]
                face_uvs_idx = mesh.face_uvs_idx[
                    sort_idx] if mesh.face_uvs_idx is not None else None
            else:
                faces = mesh.faces
                face_uvs_idx = mesh.face_uvs_idx

            for i in range(len(faces)):
                if polygon_groups is not None:
                    if group_counter < len(
                            polygon_groups
                    ) and i == polygon_groups[group_counter]:
                        obj_file.write(f"g {group_counter}\n")
                        group_counter += 1

                f = faces[i]
                if face_uvs_idx is not None:
                    f_uv = face_uvs_idx[i]
                    obj_file.write('f %d/%d %d/%d %d/%d\n' %
                                   (f[0] + 1, f_uv[0] + 1, f[1] + 1,
                                    f_uv[1] + 1, f[2] + 1, f_uv[2] + 1))
                else:
                    obj_file.write('f %d %d %d\n' %
                                   (f[0] + 1, f[1] + 1, f[2] + 1))
