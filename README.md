# Neural Octahedral Field

The implementation of preprint [Neural Octahedral Field: Octahedral prior for simultaneous smoothing and sharp edge regularization](https://arxiv.org/abs/2408.00303)

> **Note** The `release` branch contains a simplified version of the code for easier reproduction of the main results. Full experiments, ablations, and evaluation metrics are available in the `main` branch.

> **WARNING** We have only tested it on Linux (Ubuntu 22.04 and EndeavorOS).

## Environment Setup
1. Create a new environment
    ```
    conda create -n octa python=3.10 -y
    conda activate octa
    ```
2. Install [PyTorch](https://pytorch.org/get-started/locally/)
    ```bash
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    ```
3. Install [JAX](https://github.com/google/jax?tab=readme-ov-file#installation) (Jax may overwrite some of PyTorch dependencies)
    ```bash
    pip install -U "jax[cuda12]==0.6.2"
    ```
4. Then install the rest dependencies by
    ```
    pip install -r requirements.txt
    ```
5. (Optional) We also have some
[CPP bindings](https://github.com/Ankbzpx/frame-field-utils) referred as `frame_field_utils` in code. It is subject to a different License and is not required for the main results.

## Reconstruction
### JAX
Our main experiments are implemented in JAX. Run
```bash
python run_recon.py --config configs/octa_hessian.json --model /path/to/target_pointcloud.ply
```
or
```bash
python run_recon.py --config configs/octa_hessian_noisy.json --model /path/to/target_noisy_pointcloud.ply
```
based on noise level.

The extracted mesh can be found in `output` folder.

### PyTorch
We also provide a minimal [PyTorch](./pytorch) implementation.

```bash
pip install lightning
cd pytorch

python run_recon_pytorch.py --config ../configs/octa_hessian.json --model /path/to/target_pointcloud.ply

python run_recon_pytorch.py --config ../configs/octa_hessian_noisy.json --model /path/to/target_noisy_pointcloud.ply
```

## Tips
1. We only apply our regularization for on-surface samples as we assume dense point clouds as input. For regions that have insufficient observations, upsampling or techniques such as the "Dynamic Sampling" proposed in [NeurCADRecon](https://arxiv.org/pdf/2404.13420) may help improve the results, though we haven't yet evaluated this.
2. Equivalent but alternative forms of our alignment loss can be found in [loss.py](./loss.py). In particular, the `align_sh4_explicit`, introduced in our previous preprint, though much less mathematically elegant, empirically performs better at enforcing sharp edges.
3. Check out [jax2torch](https://github.com/lucidrains/jax2torch), [torch2jax](https://github.com/rdyro/torch2jax) for seamless interoperation between JAX and Pytorch. Both support backpropagation as well.
