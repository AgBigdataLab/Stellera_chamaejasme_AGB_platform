"""Boundary-conditioned deformable cross-attention for RF-DETR.

The official two sampling points per head are preserved as the interior/global
path. Two query-conditioned points initialized near the reference-box boundary
are concatenated before the same sparse deformable-attention aggregation. This
changes feature acquisition inside cross-attention instead of refining queries
with a second dense memory read after the decoder layer.
"""

from __future__ import annotations

from math import pi
from unittest.mock import patch

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.init import constant_

from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn


ARCHITECTURE_ID = "BC-DCA-RF-DETR-Seg-Large-v1"


class BoundaryConditionedMSDeformAttn(MSDeformAttn):
    """MSDeformAttn with a lightweight reference-box boundary sampling path."""

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 1,
        n_heads: int = 16,
        n_points: int = 2,
        *,
        boundary_points: int = 2,
        boundary_radii: tuple[float, ...] = (0.75, 1.0),
        boundary_residual_scale: float = 0.25,
        boundary_logit_bias: float = -4.0,
    ) -> None:
        if boundary_points <= 0:
            raise ValueError("boundary_points must be positive")
        if len(boundary_radii) != boundary_points:
            raise ValueError("boundary_radii must contain one radius per boundary point")
        if any(radius <= 0.0 for radius in boundary_radii):
            raise ValueError("boundary radii must be positive")
        if boundary_residual_scale <= 0.0:
            raise ValueError("boundary_residual_scale must be positive")

        super().__init__(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.boundary_points = boundary_points
        self.boundary_residual_scale = float(boundary_residual_scale)
        self.boundary_logit_bias = float(boundary_logit_bias)
        self.total_points = n_points + boundary_points

        self.boundary_sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * boundary_points * 2
        )
        self.boundary_attention_weights = nn.Linear(
            d_model, n_heads * n_levels * boundary_points
        )
        constant_(self.boundary_sampling_offsets.weight, 0.0)
        constant_(self.boundary_sampling_offsets.bias, 0.0)
        constant_(self.boundary_attention_weights.weight, 0.0)
        constant_(self.boundary_attention_weights.bias, boundary_logit_bias)

        angles = torch.arange(n_heads, dtype=torch.float32) * (2.0 * pi / n_heads)
        directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
        radii = torch.tensor(boundary_radii, dtype=torch.float32)
        anchors = (
            directions[:, None, None, :]
            * radii[None, None, :, None]
        ).repeat(1, n_levels, 1, 1)
        self.register_buffer("boundary_anchors", anchors, persistent=True)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Tensor | None = None,
        input_spatial_shapes_hw: list[tuple[int, int]] | None = None,
    ) -> Tensor:
        del input_level_start_index
        batch_size, len_query, _ = query.shape
        _, len_input, _ = input_flatten.shape
        if reference_points.ndim != 4 or reference_points.shape[-1] != 4:
            raise ValueError("BC-DCA requires normalized cxcywh reference boxes")
        if reference_points.shape[2] not in (1, self.n_levels):
            raise ValueError(
                f"reference-box level dimension must be 1 or {self.n_levels}"
            )
        if self._export and input_spatial_shapes_hw is None:
            raise ValueError("input_spatial_shapes_hw is required in export mode")

        if input_spatial_shapes_hw is not None:
            expected_len = sum(height * width for height, width in input_spatial_shapes_hw)
            if expected_len != len_input:
                raise ValueError("feature-map shapes do not match flattened memory")
        else:
            expected_len = int(
                (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum().item()
            )
            if expected_len != len_input:
                raise ValueError("feature-map shapes do not match flattened memory")

        if reference_points.shape[2] == 1 and self.n_levels > 1:
            reference_points = reference_points.expand(-1, -1, self.n_levels, -1)

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.transpose(1, 2).contiguous().view(
            batch_size,
            self.n_heads,
            self.d_model // self.n_heads,
            len_input,
        )

        interior_offsets = self.sampling_offsets(query).view(
            batch_size,
            len_query,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )
        interior_locations = (
            reference_points[:, :, None, :, None, :2]
            + interior_offsets
            / self.n_points
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )

        boundary_residuals = self.boundary_sampling_offsets(query).view(
            batch_size,
            len_query,
            self.n_heads,
            self.n_levels,
            self.boundary_points,
            2,
        )
        boundary_residuals = (
            torch.tanh(boundary_residuals) * self.boundary_residual_scale
        )
        anchors = self.boundary_anchors.to(dtype=query.dtype)[None, None]
        boundary_locations = (
            reference_points[:, :, None, :, None, :2]
            + (anchors + boundary_residuals)
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )
        sampling_locations = torch.cat(
            (interior_locations, boundary_locations), dim=4
        )

        interior_logits = self.attention_weights(query).view(
            batch_size,
            len_query,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )
        boundary_logits = self.boundary_attention_weights(query).view(
            batch_size,
            len_query,
            self.n_heads,
            self.n_levels,
            self.boundary_points,
        )
        attention_weights = torch.cat(
            (interior_logits, boundary_logits), dim=-1
        )
        attention_weights = F.softmax(
            attention_weights.flatten(-2), dim=-1
        ).view(
            batch_size,
            len_query,
            self.n_heads,
            self.n_levels,
            self.total_points,
        )

        if self._export:
            sampling_locations = sampling_locations.flatten(3, 4)
            attention_weights = attention_weights.flatten(3, 4)
        output = ms_deform_attn_core_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=input_spatial_shapes_hw,
        )
        return self.output_proj(output)


def install_bcdca_modules(
    model: nn.Module,
    *,
    boundary_points: int = 2,
    boundary_radii: tuple[float, ...] = (0.75, 1.0),
    boundary_residual_scale: float = 0.25,
    boundary_logit_bias: float = -4.0,
) -> nn.Module:
    """Replace every official decoder cross-attention with BC-DCA."""
    if getattr(model, "bcdca_architecture_id", None) == ARCHITECTURE_ID:
        return model
    transformer = getattr(model, "transformer", None)
    decoder = getattr(transformer, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None or len(layers) == 0:
        raise TypeError("BC-DCA requires the official RF-DETR decoder stack")

    for layer_index, layer in enumerate(layers):
        original = getattr(layer, "cross_attn", None)
        if not isinstance(original, MSDeformAttn):
            raise TypeError("decoder cross_attn is not an RF-DETR MSDeformAttn module")
        replacement = BoundaryConditionedMSDeformAttn(
            d_model=original.d_model,
            n_levels=original.n_levels,
            n_heads=original.n_heads,
            n_points=original.n_points,
            boundary_points=boundary_points,
            boundary_radii=boundary_radii,
            boundary_residual_scale=boundary_residual_scale,
            boundary_logit_bias=boundary_logit_bias,
        ).to(
            device=next(original.parameters()).device,
            dtype=next(original.parameters()).dtype,
        )
        shared_state = original.state_dict()
        incompatible = replacement.load_state_dict(shared_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected official cross-attention keys: {incompatible.unexpected_keys}"
            )
        layer.cross_attn = replacement
        layer.bcdca_layer_index = layer_index

    model.bcdca_architecture_id = ARCHITECTURE_ID
    model.bcdca_decoder_layers = len(layers)
    custom_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".cross_attn.boundary_" in name
    )
    if custom_parameters <= 0:
        raise RuntimeError("BC-DCA installation produced no custom parameters")
    model.bcdca_parameter_count = custom_parameters
    print(
        f"BC-DCA installed in {len(layers)} decoder layers; "
        f"added parameters: {custom_parameters:,}"
    )
    return model


def build_bcdca_module(
    model_config,
    train_config,
    *,
    boundary_points: int = 2,
    boundary_radii: tuple[float, ...] = (0.75, 1.0),
    boundary_residual_scale: float = 0.25,
    boundary_logit_bias: float = -4.0,
):
    """Install BC-DCA before pretrained weights and optimizer construction."""
    import rfdetr.training.module_model as module_model
    from rfdetr.training import RFDETRModelModule

    official_builder = module_model.build_model_from_config

    def custom_builder(config, training_config):
        model = official_builder(config, training_config)
        device = next(model.parameters()).device
        # Preserve the RNG used to initialize the native foreground head.
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            return install_bcdca_modules(
                model,
                boundary_points=boundary_points,
                boundary_radii=boundary_radii,
                boundary_residual_scale=boundary_residual_scale,
                boundary_logit_bias=boundary_logit_bias,
            )

    with patch.object(module_model, "build_model_from_config", custom_builder):
        module = RFDETRModelModule(model_config, train_config)

    state_keys = tuple(module.model.state_dict())
    if not any(".cross_attn.boundary_sampling_offsets." in key for key in state_keys):
        raise RuntimeError("BC-DCA parameters are absent after model construction")
    return module
