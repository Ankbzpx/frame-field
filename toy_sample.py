import numpy as np

from sh_representation import euler_to_R3, rotvec_to_R3

from icecream import ic
import polyscope as ps


def gen_toy_sample(gap, theta, offset=4, grid_size=10):
    grid_size += offset
    gap += offset - 1
    axis = np.linspace(-1, 1, grid_size)

    grid_samples = np.stack(np.meshgrid(axis, axis, axis), -1)
    sub_samples = grid_samples[-offset, ...][gap:, :].reshape(-1, 3)
    sub_samples_vn = np.zeros_like(sub_samples)
    sub_samples_vn[:, 1] = 1

    sub_samples2 = grid_samples[:, offset - 1, :][:-gap, :].reshape(-1, 3)
    sub_samples2_vn = np.zeros_like(sub_samples2)
    sub_samples2_vn[:, 0] = -1

    R = np.float64(euler_to_R3(0, 0, -np.pi / 4))
    S = np.diag(np.array([np.cos(theta / 2),
                          np.sin(theta / 2),
                          np.sqrt(2) / 2]))
    A = np.float64(rotvec_to_R3(np.array([0, 0, np.pi / 4 - theta / 2
                                         ]))) @ R @ S @ R.T

    grid_samples = grid_samples.reshape(-1, 3) @ A.T
    a = grid_samples[0]
    b = grid_samples[14 - 1]
    c = grid_samples[14**2 - 1]
    assert np.isclose(np.linalg.norm(a - b), np.linalg.norm(b - c))

    return np.vstack([sub_samples, sub_samples2]) @ A.T, np.vstack([
        sub_samples_vn, sub_samples2_vn @ np.float64(
            rotvec_to_R3(np.array([0, 0, -np.pi / 2 + theta])))
    ]), grid_samples


if __name__ == '__main__':
    for gap in [1, 2, 3, 4]:
        for theta in [150, 135, 120, 90, 60, 45, 30]:
            samples_sup, samples_vn_sup, samples_interp = gen_toy_sample(
                gap, np.deg2rad(theta))

            ps.init()
            ps.register_point_cloud('samples_sup', samples_sup)
            ps.show()
            exit()

            # Add small rotation to avoid axis aligned bias
            R = euler_to_R3(np.pi / 6, np.pi / 3, np.pi / 4)
            samples_sup = samples_sup @ R.T
            # R^(-T) = R
            samples_vn_sup = samples_vn_sup @ R.T
            samples_interp = samples_interp @ R.T

            centroid = np.mean(samples_interp, axis=0, keepdims=True)
            V_max = np.amax(samples_interp)
            V_min = np.amin(samples_interp)

            samples_sup -= centroid
            samples_sup = (samples_sup - V_min) / (V_max - V_min)
            samples_interp -= centroid
            samples_interp = (samples_interp - V_min) / (V_max - V_min)

            # [-0.95, 0.95]
            samples_sup -= 0.5
            samples_sup *= 1.9
            samples_interp -= 0.5
            samples_interp *= 1.9

            np.savez(f"data/toy/crease_{gap}_{theta}.npz",
                     samples_sup=samples_sup,
                     samples_vn_sup=samples_vn_sup,
                     samples_interp=samples_interp)
