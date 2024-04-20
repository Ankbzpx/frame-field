## A Benchmark for Surface Reconstruction

Run Synthetic Scanning from the working version
https://github.com/fwilliams/surface-reconstruction-benchmark using docker

```
# Build base image first
docker build -t "srb:latest" --file srb.Dockerfile .

docker build -t "mesh_to_implicit:latest" --file mesh_to_implicit.Dockerfile .

docker build -t "isosurface:latest" --file isosurface.Dockerfile .

docker build -t "pc_generator:latest" --file pc_generator.Dockerfile .

docker build -t "run_sampler:latest" --file run_sampler.Dockerfile .
```

Sample commands
```
# mkdir data/implicit
docker run --rm -v "$HOME/frame-field/data:/data" mesh_to_implicit /data/mesh/fandisk.ply /data/implicit/fandisk.mpu 6 0.01 1.0

docker run --rm -v "$HOME/frame-field/data:/data" isosurface /data/implicit/fandisk.mpu 256 /data/implicit/fandisk.obj

# mkdir data/recon
docker run --rm -v "$HOME/frame-field/data:/data" pc_generator /data/implicit/fandisk.mpu /data/recon/fandisk res 350 scans 8 min_range 75 max_range 115 additive_noise 0.6 registration_noise 0.4 peak_threshold 0.2

docker run --rm -v "$HOME/frame-field/data:/data" run_sampler /data/recon/fandisk
```
