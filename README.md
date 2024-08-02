# Neural Octahedral Field

The implementation of preprint [Neural Octahedral Field: Octahedral prior for simultaneous smoothing and sharp edge regularization](https://arxiv.org/abs/2408.00303)

> **WARNING** This is a research repo with limited code quality. We have only tested it on Linux (Ubuntu 22.04 and EndeavorOS).

## Environment Setup
We first create a new environment
```
conda create -n octa python=3.10 -y
conda activate octa
```
Follow instructions to install [JAX](https://github.com/google/jax?tab=readme-ov-file#installation) and [PyTorch (cpu)](https://pytorch.org/get-started/locally/), e.g.
```
pip install -U "jax[cuda12]"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Then install [libigl](https://github.com/libigl/libigl-python-bindings) and the rest dependencies by
```
python -m pip install libigl
pip install -r requirements.txt
```
> (Optional) We also have some
[CPP bindings](https://github.com/Ankbzpx/frame-field-utils) referred as `frame_field_utils` in code. It is subject to difference Licenses and is not required for the main results

## Reconstruction
Run
```
python run_recon.py --config configs/octa_hessian.json --model /path/to/target_pointcloud.ply
```
or
```
python run_recon.py --config configs/octa_hessian_noisy.json --model /path/to/target_noisy_pointcloud.ply
```
based on noise level

See [metric](./metric) folder for datasets and compared methods
