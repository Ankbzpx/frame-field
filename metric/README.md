# Metric
We store our dataset and results in hierarchy like
```
dataset
    |
    |---p2s
    |   |-abc
    |   |  |---1e-2
    |   |  |---2e-3
    |   |  |---gt
    |   |-thingi10k
    |       |---1e-2
    |       |---2e-3
    |       |---gt
    |
    |---octa_results
            |---method_1
            |     |
            |     |---abc_1e-2
            |     |---abc_2e-3
            |     |---thingi10k_1e-2
            |     |---thingi10k_2e-3
           ...
```
For methods output pointcloud, we additional construct surface using [SPSR](https://www.cs.jhu.edu/~misha/MyPapers/ToG13.pdf) or [Advancing Front](https://doc.cgal.org/latest/Advancing_front_surface_reconstruction/index.html) for visualization of a similar hierarchy (e.g. `method_1_viz`)

We compute our metrics using
```
python compute_metrics.py
python collect_metrics.py
```

## Methods
### [DiGS](https://chumbyte.github.io/DiGS-Site/)
> Referred as `digs`

We use script `surface_reconstruction/scripts/run_surf_recon_exp.sh` in
[repo](https://github.com/Chumbyte/DiGS) of commit `
44bcc8`, for both noise levels

### [Edge Aware Resampling](https://vcc.tech/research/2013/EAR)
> Referred as `EAR`

We use the implementation in [CGAL](https://doc.cgal.org/latest/Point_set_processing_3/index.html) of version 5.6.1, same config for both noise levels

- Bilateral smooth
    ```
    k = 120
    sharpness_angle = 25
    iter_number = 3
    ```
- Resampling
    ```
    edge_sensitivity = 0.3
    number_of_output_points = points_size * 4
    ```

### [Algebraic Point Set Surfaces](https://www.labri.fr/perso/guenneba/docs/APSS_sig07.pdf)
> Referred as `APSS`

We use the implementation in [MeshLab](https://www.meshlab.net/) of version 2023.12, for both noise levels

### [Graph Laplacian Regularization](https://arxiv.org/abs/1803.07252)
> Referred as `graph_laplacian`

We use implementation in [repo](https://github.com/jzengust/ldmm_graph_laplacian_pointcloud_denoise) of commit `f8c96e`, and tune w.r.t. noise levels
- 2e-3: `lambda_b = 4`
- 1e-2: `lambda_b = 7`

### [Screened Poisson Surface Reconstruction](https://www.cs.jhu.edu/~misha/MyPapers/ToG13.pdf)
> Referred as `SPR`

We use the implementation in [MeshLab](https://www.meshlab.net/) of version 2023.12, for both noise levels

### [Neural Kernel Surface Reconstruction](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Neural_Kernel_Surface_Reconstruction_CVPR_2023_paper.pdf)
> Referred as `nksr`

We use the 'kitchen' pre-trained model `ks.pth` in
[repo](https://github.com/nv-tlabs/NKSR) of commit `0d4e36`, with voxel size `1e-2`, for both noise levels

We also tried `p2s.pth`, but found it inferior metric-wise

### [Robust Pointset Denoising of Piecewise-Smooth Surfaces through Line Processes](https://jiongchen.github.io/files/lineproc-paper.pdf)
> Referred as `line_processing`

We use implementation in
[repo](https://github.com/kwwei/line-process-pointset-denoising) of commit `49680`, and tune w.r.t. noise levels

```
w_a=1
w_b=5000
mu_smooth=30.0
smooth_comparison="NULL"
lp_threshold=0.3
reconstruction_method=2
need_s=1
mu_fit=5e-9
max_smooth_iter=5
needs_post_process=0
```
- 2e-3
    ```
    w_c: 1
    k_neighbor: 50
    ```
- 1e-2
    ```
    w_c: 2
    k_neighbor: 150
    ```

### [SIREN](https://arxiv.org/abs/2006.09661)
> Referred as `siren`

Use `configs/siren` for both noisy level

### Ours
`ours_hessian_5`
- 2e-3: `octa_hessian_5`
- 1e-2: `octa_hessian_noisy_5`

`ours_hessian_10`
- 2e-3: `octa_hessian`
- 1e-2: `octa_hessian_noisy`

`ours_digs_5`
- 2e-3: `configs/octa_digs_5`
- 1e-2: `configs/octa_digs_noisy_5`

`ours_digs_10`
- 2e-3: `configs/octa_digs`
- 1e-2: `configs/octa_digs_noisy`

### [Neural-Singular-Hessian](https://dl.acm.org/doi/10.1145/3618311)
> Referred as `neural_singular_hessian`

We use script `surface_reconstruction
/run_sdf_recon.py` in [repo](https://github.com/bearprin/Neural-Singular-Hessian) with commit `ca7da0`, for both noise levels

### [SALD](https://arxiv.org/abs/2006.05400)
> Referred as `SALD`

We adapt code in [repo](https://github.com/matanatz/SALD) with commit `
6402b7`,
for both noise levels
- Use PCA normals as manifold normals
- Approximate non-manifold normals with average direction to closed input cloud, with `k=51`

### [IterativePFN](https://arxiv.org/abs/2304.01529)
> Referred as `IterativePFN`

We use pre-trained model provided in [repo](https://github.com/ddsediri/IterativePFN) of commit `79efe4`, for both noise levels

### [RFEPS](https://xrvitd.github.io/Projects/RFEPS/index.html)
> Referred as `RFEPS`

We use the implementation in [repo](https://github.com/Xrvitd/RFEPS) of commit `d2b915`, for both noise levels

### [NeurCADRecon](https://arxiv.org/abs/2404.13420)
> Referred as `NeurCAD`

We use script `surface_reconstruction/train_surface_reconstruction.py` in
[repo](https://github.com/QiujieDong/NeurCADRecon) of commit `794a10`, for both noise levels
