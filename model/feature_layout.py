from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import torch


FeatureMode = Literal["raw6", "net4"]


@dataclass(frozen=True)
class FeatureLayout:
    mode: FeatureMode
    num_features: int
    power_feature_count: int
    voltage_index: int
    angle_index: int


def get_feature_layout(feature_mode: str) -> FeatureLayout:
    if feature_mode == "raw6":
        return FeatureLayout(
            mode="raw6",
            num_features=6,
            power_feature_count=4,
            voltage_index=4,
            angle_index=5,
        )
    if feature_mode == "net4":
        return FeatureLayout(
            mode="net4",
            num_features=4,
            power_feature_count=2,
            voltage_index=2,
            angle_index=3,
        )
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def infer_feature_mode_from_num_features(num_features: int) -> FeatureMode:
    if num_features == 6:
        return "raw6"
    if num_features == 4:
        return "net4"
    raise ValueError(f"Cannot infer feature mode from num_features={num_features}")


def validate_feature_mode_num_features(feature_mode: str, num_features: int) -> None:
    expected_num_features = get_feature_layout(feature_mode).num_features
    if num_features != expected_num_features:
        raise ValueError(
            f"Feature mode {feature_mode} expects {expected_num_features} features, got {num_features}"
        )


def convert_node_features_to_mode(
    node_features: Union[np.ndarray, torch.Tensor],
    feature_mode: str,
):
    layout = get_feature_layout(feature_mode)
    current_mode = infer_feature_mode_from_num_features(int(node_features.shape[-1]))

    if current_mode == feature_mode:
        return node_features
    if current_mode != "raw6" or feature_mode != "net4":
        raise ValueError(f"Unsupported feature conversion: {current_mode} -> {feature_mode}")

    p_net = node_features[..., 0] - node_features[..., 2]
    q_net = node_features[..., 1] - node_features[..., 3]
    voltage = node_features[..., 4]
    angle = node_features[..., 5]

    if isinstance(node_features, torch.Tensor):
        converted = torch.stack([p_net, q_net, voltage, angle], dim=-1)
    else:
        converted = np.stack([p_net, q_net, voltage, angle], axis=-1)

    validate_feature_mode_num_features(layout.mode, int(converted.shape[-1]))
    return converted
