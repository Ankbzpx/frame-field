import argparse
import json
import os

from common import (
    aabb_compute,
    filter_components,
    normalize,
    ps_register_curve_network,
    rm_unref_vertices,
    Timer,
    vis_oct_field,
    voxel_tet_from_grid_scale,
    write_triangle_mesh_VC,
)
from config import Config
from config_utils import config_latent, config_model, load_sdf
import model_jax
from sh_representation import (
    proj_sh4_sdp,
    proj_sh4_to_R3,
    proj_sh4_to_rotvec,
    project_n,
    R3_to_repvec,
    R3_to_sh4_zonal,
    rot6d_to_R3,
    rot6d_to_sh4_zonal,
    rotvec_n_to_z,
    rotvec_to_R3,
    rotvec_to_R9,
    rotvec_to_sh4,
)

import equinox as eqx
import igl
import jax
from jax import jit, numpy as jnp, vmap
import numpy as np
from skimage.measure import marching_cubes

from icecream import ic
import polyscope as ps


# https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
# fmt: off
turbo_colormap_data = np.array([[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]])
# fmt: on


# infer: R^3 -> R
def voxel_infer(
    infer,
    grid_res=512,
    grid_bl=np.array([-1.0, -1.0, -1.0]),
    grid_tr=np.array([1.0, 1.0, 1.0]),
    group_size_mul=1,
    out_dim=1,
):
    # Smaller batch is somehow faster
    group_size = group_size_mul * grid_res**2
    iter_size = grid_res**3 // group_size

    # Cannot pass jitted function as argument to another jitted function
    @jit
    def infer_scalar():
        # For consistency with partition, we ignore the endpoint
        idx_x = jnp.linspace(grid_bl[0], grid_tr[0], grid_res, endpoint=False)
        idx_y = jnp.linspace(grid_bl[1], grid_tr[1], grid_res, endpoint=False)
        idx_z = jnp.linspace(grid_bl[2], grid_tr[2], grid_res, endpoint=False)
        grid = jnp.stack(jnp.meshgrid(idx_x, idx_y, idx_z), -1)

        query_data = {
            "grid": grid.reshape(iter_size, group_size, 3),
            "val": jnp.zeros((iter_size, group_size, out_dim)),
        }

        @jit
        def body_func(i, query_data):
            val = infer(query_data["grid"][i]).reshape(-1, out_dim)
            query_data["val"] = query_data["val"].at[i].set(val)
            return query_data

        query_data = jax.lax.fori_loop(0, iter_size, body_func, query_data)
        return query_data["val"].reshape(grid_res, grid_res, grid_res, out_dim), grid

    return infer_scalar()


# infer: R^3 -> R (sdf)
def extract_surface(infer, grid_res=512, grid_min=-1.0, grid_max=1.0, iso=0.0):
    grid_max_res = 512

    if grid_res > grid_max_res:
        # Have to partition
        div = int(np.ceil(grid_res / grid_max_res))
        interval = (grid_max - grid_min) / div

        part_idx = np.arange(div)
        part_offsets = np.stack(np.meshgrid(part_idx, part_idx, part_idx), -1).reshape(
            -1, 3
        )

        part_bl = grid_min + part_offsets * interval
        part_tr = grid_min + (part_offsets + 1) * interval

        block_list = []
        for i in range(div**3):
            sdf, _ = voxel_infer(
                infer, grid_max_res, grid_bl=part_bl[i], grid_tr=part_tr[i]
            )
            block_list.append(np.array(sdf[..., 0]))

        sdf_np = np.stack(block_list).reshape(
            div, div, div, grid_max_res, grid_max_res, grid_max_res
        )
        sdf_np = np.transpose(sdf_np, (0, 3, 1, 4, 2, 5)).reshape(
            grid_res, grid_res, grid_res
        )
    else:
        sdf, _ = voxel_infer(
            infer,
            grid_res,
            grid_bl=np.array([grid_min, grid_min, grid_min]),
            grid_tr=np.array([grid_max, grid_max, grid_max]),
        )
        # This step is surprising slow, gpu to cpu memory copy?
        sdf_np = np.array(sdf[..., 0])

    sdf_np = np.swapaxes(sdf_np, 0, 1)
    spacing = 1.0 / grid_res
    # It outputs inverse VN, even with gradient_direction set to ascent
    V, F, VN_inv, _ = marching_cubes(sdf_np, iso, spacing=(spacing, spacing, spacing))
    dim = grid_max - grid_min
    V = dim * (V - np.abs(grid_min) / dim)
    return V, F, -VN_inv


# Reduce face count to speed up visualization
# TODO: Use edge collapsing like one in Instant meshes
def meshlab_edge_collapse(save_path, V, F, num_faces):
    import pymeshlab

    m = pymeshlab.Mesh(V, F)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "mesh")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=num_faces)

    # FIXME: read back instead of saving to file
    ms.save_current_mesh(save_path)
    V, F = igl.read_triangle_mesh(save_path)
    return V, F


def batch_call(
    func, input, num_out_args=1, out_map_func=lambda x: [x], group_size=256**2
):
    n_iters = len(input) // group_size

    if n_iters == 0:
        output = func(input)
        output = out_map_func(output)
    else:
        output = {}
        for i in range(num_out_args):
            output[i] = None

        input_splits = jnp.array_split(input, n_iters)
        for input_batch in input_splits:
            output_ = func(input_batch)
            output_ = out_map_func(output_)

            for i in range(num_out_args):
                output[i] = (
                    output_[i]
                    if output[i] is None
                    else jnp.concatenate([output[i], output_[i]])
                )

        output = list(output.values())

    if num_out_args == 1:
        output = output[0]

    return output


def eval(
    cfg: Config,
    model: model_jax.MLP,
    latent,
    grid_res=512,
    vis_singularity=False,
    vis_mc=False,
    vis_smooth=False,
    vis_flowline=False,
    save_octa=False,
    trace_flowline=False,
    miq=False,
    interp_tag="",
    udf=False,
    dcudf=False,
):
    # Map network output to sh4 parameterization
    if cfg.loss_cfg.rot6d:
        param_func = rot6d_to_sh4_zonal
        proj_func = vmap(rot6d_to_R3)
    elif cfg.loss_cfg.rotvec:
        param_func = rotvec_to_sh4
        proj_func = vmap(rotvec_to_R3)
    else:
        param_func = lambda x: x
        proj_func = proj_sh4_to_R3

    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)

    @jit
    def infer_grad(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model.call_grad(x, z)

    @jit
    def infer_smoothness(x):
        z = latent[None, ...].repeat(len(x), 0)
        jac, _ = model.call_jac_param(x, z, param_func)
        return vmap(jnp.linalg.norm, in_axes=(0, None))(jac, "f")

    timer = Timer()

    if udf:
        grid_res = 256
        iso = 6e-3

        sdf_data = load_sdf(cfg.sdf_paths[0])
        sur_sample = sdf_data["samples_on_sur"]
        pc_center, pc_scale, _ = aabb_compute(sur_sample)

        # Compute original bound in normalized space
        bound_max = sur_sample.max(0, keepdims=True)
        bound_min = sur_sample.min(0, keepdims=True)
        bound_max = (bound_max - pc_center) / pc_scale
        bound_min = (bound_min - pc_center) / pc_scale

        @jit
        def infer_udf(x):
            df = infer(x)[:, 0]
            df = jnp.clip(df / 1000, min=1e-10)
            return jnp.sqrt(df)

        if dcudf:
            from dcudf.mesh_extraction import dcudf as DCUDF

            from jax2torch import jax2torch

            infer_udf_torch = jax2torch(infer_udf)
            ms = DCUDF(
                infer_udf_torch,
                resolution=grid_res,
                threshold=iso,
                laplacian_weight=200,
                learning_rate=5e-5,
                report_freq=200,
                bound_min=bound_min[0],
                bound_max=bound_max[0],
                is_cut=False,
            ).optimize()
            V = ms.vertices
            F = ms.faces
        else:
            V, F, _ = extract_surface(infer_udf, grid_res=grid_res, iso=iso)
    else:
        infer_sdf = lambda x: infer(x)[:, 0]
        V, F, _ = extract_surface(infer_sdf, grid_res=grid_res)

    timer.log("Extract surface")

    if vis_smooth:
        smoothness = batch_call(infer_smoothness, V)
        s_max = smoothness.max()
        s_min = smoothness.min()
        smoothness = (smoothness - s_min) / (s_max - s_min)
        qua_idx = (smoothness * len(turbo_colormap_data)).astype(np.int32)
        qua_idx = np.clip(qua_idx, 0, len(turbo_colormap_data) - 1)
        smoothness_color = turbo_colormap_data[qua_idx]
        timer.log("Infer smoothness")

    if vis_singularity:
        import frame_field_utils

        V_tet, T = voxel_tet_from_grid_scale(16, 1)

        def out_map_func(out):
            (sdf_, aux_), VN_ = out
            return sdf_, aux_, VN_

        sdf, aux, VN = batch_call(infer_grad, V_tet, 3, out_map_func)

        # V_tet, T, V_id = frame_field_utils.tet_reduce(V_tet, VN, sdf < 0, T)
        # aux = aux[V_id]
        # VN = VN[V_id]
        sh4 = vmap(param_func)(aux)

        timer.log("Extract parameterization")

        sh4 = proj_sh4_sdp(sh4)
        sh4_bary = sh4[T].mean(axis=1)
        Rs_bary = proj_sh4_to_R3(sh4_bary)

        timer.log("Project and interpolate SH4")

        TT, TTi = igl.tet_tet_adjacency(T)
        uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum, E2uE, E2T = (
            frame_field_utils.tet_edge_one_ring(T, TT)
        )
        uE_singularity_mask = frame_field_utils.tet_frame_singularity(
            uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum, Rs_bary
        )
        uE_singular = uE[uE_singularity_mask]

        timer.log("Compute singularity")

        F_b = igl.boundary_facets(T)
        F_b = np.stack([F_b[:, 2], F_b[:, 1], F_b[:, 0]], -1)

        V_b, F_b = rm_unref_vertices(V_tet, F_b)
        # igl.write_triangle_mesh(f"{cfg.out_dir}/{cfg.name}_tet_bound.obj",
        #                         np.float64(V_b), F_b)

        # V_uE, uE_singular = rm_unref_vertices(V_tet, uE_singular)
        # data = {
        #     'V': V_uE.reshape(-1,).tolist(),
        #     'uE': uE_singular.reshape(-1,).tolist()
        # }
        # with open(f'{cfg.out_dir}/{cfg.name}.json', 'w') as f:
        #     json.dump(data, f)

        # exit()

        ps.init()
        ps.register_surface_mesh("tet boundary", V_b, F_b, enabled=False)
        ps.register_surface_mesh("mc", V, F)
        if uE_singularity_mask.sum() > 0:
            ps_register_curve_network("singularity", V_tet, uE_singular)
        ps.show()

        param_path = os.path.join(f"{cfg.out_dir}/{cfg.name}.npz")
        np.savez(param_path, V=V_tet, T=T, sh4=sh4, sdf=sdf)

        exit()

    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)

    # Recovery input scale
    # TODO: support latent
    sdf_data = load_sdf(cfg.sdf_paths[0])
    sur_sample = sdf_data["samples_on_sur"]
    pc_center, pc_scale, _ = aabb_compute(sur_sample)

    save_name = f"{cfg.name}_{interp_tag}" if interp_tag != "" else f"{cfg.name}"

    # Octahedral field
    if save_octa and len(cfg.mlp_cfgs) > 1:
        aux = batch_call(infer, V)[:, 1:]
        sh4 = param_func(aux)

        if cfg.loss_cfg.xy_scale != 1:
            sh4 = proj_sh4_sdp(sh4)

        print(f"SH4 norm {vmap(jnp.linalg.norm)(sh4).mean()}")
        Rs = proj_func(sh4)

        timer.log("Infer octahedral frames")

        V_vis_sup, F_vis_sup = vis_oct_field(Rs, V, 0.64 / grid_res)
        V_vis_sup = V_vis_sup * pc_scale + pc_center
        igl.write_triangle_mesh(
            os.path.join(cfg.out_dir, f"{save_name}_octa.obj"), V_vis_sup, F_vis_sup
        )

    V = V * pc_scale + pc_center
    if vis_smooth:
        write_triangle_mesh_VC(
            os.path.join(cfg.out_dir, f"{save_name}.obj"), V, F, smoothness_color
        )
    else:
        igl.write_triangle_mesh(os.path.join(cfg.out_dir, f"{save_name}.obj"), V, F)

    if miq:
        import frame_field_utils

        aux = infer(V)[:, 1:]
        sh4 = param_func(aux)
        sh4 = proj_sh4_sdp(sh4)

        FN = igl.per_face_normals(V, F, np.float64([0, 1, 0]))

        sh4 = sh4[F].mean(1)
        Rs = proj_sh4_to_R3(sh4)

        Q = vmap(R3_to_repvec)(Rs, FN)

        UV, FUV = frame_field_utils.miq(
            np.float64(V), F, np.float64(Q), gradient_size=75
        )

        timer.log("MIQ")

        from mesh_helper import OBJMesh, write_obj

        mesh = OBJMesh(V, F)
        mesh.uvs = UV
        mesh.face_uvs_idx = FUV

        write_obj(f"{cfg.out_dir}/{cfg.name}_param.obj", mesh)

        exit()

    if vis_mc:
        ps.init()
        mesh = ps.register_surface_mesh(f"{cfg.name}", V, F)
        # if len(cfg.mlp_cfgs) > 1:
        #     ps.register_surface_mesh('Oct frames supervise', V_vis_sup,
        #                              F_vis_sup)

        # pc = ps.register_point_cloud('sur_sample', sur_sample, radius=1e-4)
        # pc.add_vector_quantity('sur_normal', sur_normal, enabled=True)
        ps.show()
        exit()

    timer.reset()

    if trace_flowline:
        import frame_field_utils

        # Project on isosurface
        (sdf, _), VN = infer_grad(V)
        VN = vmap(normalize)(VN)
        V = V - sdf[:, None] * VN
        V = np.array(V)

        timer.log("Project SDF")

        (_, aux), VN = infer_grad(V)
        sh4 = vmap(param_func)(aux)

        if cfg.loss_cfg.xy_scale != 1:
            sh4 = proj_sh4_sdp(sh4)

        print(f"SH4 norm {vmap(jnp.linalg.norm)(sh4).mean()}")

        L = igl.cotmatrix(V, F)
        smoothness = np.trace(sh4.T @ -L @ sh4)
        print(f"Smoothness {smoothness}")

        Rs = proj_func(aux)
        timer.log("Project SO(3)")

        timer.reset()

        Q = vmap(R3_to_repvec)(Rs, VN)

        timer.log("Project to representation vectors")

        V_vis, F_vis, VC_vis = frame_field_utils.trace(V, F, VN, Q, 4000)

        timer.log("Trace flowlines")

        if vis_flowline:
            ps.init()
            mesh = ps.register_surface_mesh("mesh", V, F)
            mesh.add_vector_quantity("VN", VN)
            flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
            flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
            ps.show()

        write_triangle_mesh_VC(
            f"{cfg.out_dir}/{cfg.name}_{interp_tag}stroke.obj", V_vis, F_vis, VC_vis
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument(
        "--interp", type=str, default="0_1_0", help="Interpolation progress"
    )
    parser.add_argument(
        "--vis_singularity",
        action="store_true",
        help="Visualize octahedron singularity",
    )
    parser.add_argument("--vis_mc", action="store_true", help="Visualize MC mesh only")
    parser.add_argument("--udf", action="store_true", help="Extract UDF")
    parser.add_argument("--dcudf", action="store_true", help="Extract using DCUDF")
    parser.add_argument(
        "--vis_smooth", action="store_true", help="Visualize smoothness"
    )
    parser.add_argument(
        "--vis_flowline", action="store_true", help="Visualize flowline"
    )
    parser.add_argument("--output", type=str, default="output", help="Output folder")
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split("/")[-1].split(".")[0]
    cfg.out_dir = args.output

    latents, latent_dim = config_latent(cfg)
    tokens = args.interp.split("_")
    # Interpolate latent
    i = int(tokens[0])
    j = int(tokens[1])
    t = float(tokens[2])
    latent = (1 - t) * latents[i] + t * latents[j]

    model_key = jax.random.PRNGKey(0)
    model = config_model(cfg, model_key, latent_dim)
    model: model_jax.MLP = eqx.tree_deserialise_leaves(
        os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model
    )

    eval(
        cfg,
        model,
        latent,
        vis_singularity=args.vis_singularity,
        vis_mc=args.vis_mc,
        vis_smooth=args.vis_smooth,
        vis_flowline=args.vis_flowline,
        udf=args.udf,
        dcudf=args.dcudf,
    )
