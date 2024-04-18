from dataclasses import dataclass, field, asdict


@dataclass
class MLPConfig:
    in_features: int = 3
    hidden_features: int = 256
    hidden_layers: int = 4
    out_features: int = 1
    activation: str = 'elu'
    input_scale: float = 1    # spatial scaling


@dataclass
class TrainingConfig:
    lr: float = 5e-5
    warmup_steps: int = 100
    n_epochs: int = 1
    n_steps: int = 10000    # per epoch
    n_samples: int = 15000    # per step
    plot_every: int = 500
    seed: int = 2139028991    # 1111111011111101111110111111111
    close_sample_sigma: float = 5e-2
    close_sample_ratio: float = 0.25
    n_input_samples: int = 10000


@dataclass(frozen=True)
class LossConfig:
    on_sur: float = 7e3
    off_sur: float = 6e2
    normal: float = 1e2
    eikonal: float = 5e1

    align: float = 0    # sh4
    unit_norm: float = 1    # sh4 norm
    lip: float = 0
    smooth: float = 0    # sh4 jac
    regularize: float = 0    # off surface
    xy_scale: float = 1    # actually sqrt(xy_scale)
    hessian: float = 0

    rotvec: bool = False
    rot6d: bool = False
    explicit_basis: bool = False

    grid: bool = False


@dataclass
class Config:
    sdf_paths: list[str]
    training: TrainingConfig = TrainingConfig()
    loss_cfg: LossConfig = LossConfig()

    mlp_types: list[str] = field(default_factory=lambda: ['StandardMLP'])
    mlps: list[MLPConfig] = field(default_factory=lambda: [MLPConfig()])

    @property
    def mlp_cfgs(self) -> list[dict]:
        return [asdict(mlp) for mlp in self.mlps]

    def __post_init__(self):
        n_mlp_types = len(self.mlp_types)
        n_mlps = len(self.mlps)

        if n_mlp_types != n_mlps:
            assert n_mlp_types == 1 or n_mlps == 1

        self.mlp_types = self.mlp_types * (max(n_mlp_types, n_mlps) //
                                           n_mlp_types)
        self.mlps = self.mlps * (max(n_mlp_types, n_mlps) // n_mlps)

        self.mlps = [
            MLPConfig(**mlp) if not isinstance(mlp, MLPConfig) else mlp
            for mlp in self.mlps
        ]

        if not isinstance(self.training, TrainingConfig):
            self.training = TrainingConfig(**self.training)

        if not isinstance(self.loss_cfg, LossConfig):
            self.loss_cfg = LossConfig(**self.loss_cfg)
