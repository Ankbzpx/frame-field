from dataclasses import dataclass, field, asdict


@dataclass
class MLPConfig:
    in_features: int = 3
    hidden_features: int = 256
    hidden_layers: int = 4
    out_features: int = 1
    activation: str = 'elu'
    sdf_mlp_type: str = 'Siren'
    aux_mlp_type: str = 'Siren'


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    lr_peak: float = 5e-4
    n_epochs: int = 1
    n_steps: int = 100000    # per epoch
    n_samples: int = 1024    # per step
    plot_every: int = 1000
    seed: int = 2139028991    # 1111111011111101111110111111111


@dataclass(frozen=True)
class LossConfig:
    on_sur: float = 3e3
    off_sur: float = 1e2
    normal: float = 1e2
    eikonal: float = 5e1

    # align & twist must match
    align: float = 1e2
    twist: float = 1e2
    lip: float = 0
    smooth: float = 0


@dataclass
class Config:
    sdf_path: str
    mlp_type: str = "MLP"
    mlp: MLPConfig = MLPConfig()
    training: TrainingConfig = TrainingConfig()
    loss_cfg: LossConfig = LossConfig()

    @property
    def mlp_cfg(self) -> dict:
        return asdict(self.mlp)

    def __post_init__(self):
        if not isinstance(self.mlp, MLPConfig):
            self.mlp = MLPConfig(**self.mlp)

        if not isinstance(self.training, TrainingConfig):
            self.training = TrainingConfig(**self.training)

        if not isinstance(self.loss_cfg, LossConfig):
            self.loss_cfg = LossConfig(**self.loss_cfg)
