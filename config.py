import os
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass(frozen=True)
class ProjectConfig:
    dataset_path: str = "dataset_prepared"
    model_path: str = "spina_bifida_model.keras"
    architecture: str = "vgg16"
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-4
    seed: int = 42
    run_root: str = "runs"
    prediction_uncertain_threshold: float = 0.60
    use_class_weights: bool = True


def build_run_dir(root: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def config_to_dict(config: ProjectConfig) -> dict:
    cfg = asdict(config)
    cfg["image_size"] = list(config.image_size)
    return cfg
