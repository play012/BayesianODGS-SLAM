# Omnidirectional Gaussian Rasterization with Camera Pose Jacobians and Per-Splat Variance

A CUDA rasterizer that combines the two parent implementations in this
workspace:

* [`omni-gaussian-rasterization-w-pose`](../omni-gaussian-rasterization-w-pose) —
  omnidirectional (equirectangular) Gaussian splatting, spherical covariance,
  omnidirectional depth, and `lat`/`lon`/`psi` auxiliaries for densification.
* [`diff-gaussian-rasterization-w-pose-var`](../diff-gaussian-rasterization-w-pose-var) —
  per-splat appearance variance, variance-map rendering, and analytic SE(3)
  camera-pose Jacobians.

The result is a single differentiable rasterizer that supports, at the same
time:

- **360° spherical projection** using longitude/latitude and omnidirectional
  radial depth.
- **Spherical EWA covariance** (`W * T * SQJ`) and its backward pass.
- **Per-splat RGB variance** and a per-pixel variance image computed with the
  law of total variance, including a fully analytic backward pass for the
  variance parameters, colors, and opacities.
- **Analytic SE(3) pose Jacobians** for the left-perturbed world-to-camera pose,
  returned as separate translation (`rho`) and rotation (`theta`) gradients.
- The original `lat`, `lon`, `psi`, per-splat hit count, and opacity/coverage
  outputs used by omnidirectional SLAM pipelines.

## Install

CUDA and PyTorch are required.

```bash
cd omni-gaussian-rasterization-w-pose-var
pip install -e .
```

The package is imported as:

```python
from omni_var_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
```

## API

### Rasterization settings

```python
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    mapping_mode: bool = True
    tracking_mode: bool = True
    debug: bool = False
```

`mapping_mode` enables the map/geometry gradients (means, covariance, scale,
rotation, SH). `tracking_mode` enables the SE(3) pose Jacobian. Both are enabled
by default.

### Forward call

```python
color, depth, opacity, radii, psi, lat, lon, n_touched, variance = \
    rasterizer(
        means3D,
        means2D,
        opacities,
        shs=shs,                 # or colors_precomp=...
        scales=scales,
        rotations=rotations,     # or cov3D_precomp=...
        theta=theta,
        rho=rho,
        color_variance=sigma2,   # [P, 3], optional
        return_variance=True,
    )
```

Shape conventions:

| output | shape | meaning |
| --- | --- | --- |
| `color` | `[3, H, W]` | rendered RGB |
| `depth` | `[H, W]` | omnidirectional radial depth |
| `opacity` | `[H, W]` | accumulated coverage `1 - T` |
| `radii` | `[P]` | screen-space radius used for culling |
| `psi` | `[P]` | screen-space covariance orientation |
| `lat`, `lon` | `[P]` | spherical coordinates used for densification |
| `n_touched` | `[P]` | number of pixels each splat contributed to |
| `variance` | `[3, H, W]` or `[]` | per-pixel RGB variance map |

`means2D` is accepted for compatibility with the parent APIs; the forward pass
does not consume its value. When its shape matches the internal `[P, 3]`
projection gradient, that gradient is returned through it, mirroring the parent
implementations.

Minimal training step, including pose gradients:

```python
theta = torch.zeros(3, device="cuda", requires_grad=True)
rho = torch.zeros(3, device="cuda", requires_grad=True)

color, depth, opacity, radii, psi, lat, lon, n_touched, variance = rasterizer(
    means3D, means2D, opacities,
    shs=shs, scales=scales, rotations=rotations,
    theta=theta, rho=rho,
    color_variance=sigma2, return_variance=True,
)

loss = (color - target).abs().mean() + 0.01 * variance.mean()
loss.backward()
# theta.grad, rho.grad now hold dL/dtheta and dL/drho.
```

`theta` and `rho` are optional tensors that simply receive the analytic pose
gradients. The caller remains responsible for constructing `viewmatrix` from the
pose, as in the parent implementations.

### Backward / pose gradients

`theta` is the rotation part and `rho` is the translation part of a left
perturbation of the world-to-camera pose:

```text
T_CW <- Exp([rho, theta]) T_CW
p_C   = R m + t
```

The backward pass returns `dL/dtheta` and `dL/drho` with the same shape as the
provided `theta`/`rho` tensors (e.g. `[3]` or `[1, 3]`), or `None` when the
corresponding tensor was omitted. Internally the per-Gaussian pose Jacobian is
accumulated in `dL_dtau[P, 6]` and reduced over all Gaussians. Both the
projection/depth path and the spherical covariance path contribute.

For the covariance path the Jacobian contains both dependencies of
`J_o = W * T(lat, lon) * SQJ(dist)` on the pose:

- the camera-space centre `p_C = R m + t` enters `lat`, `lon` and `dist`
  (`dL/dtheta = p_C x dL/dp_C`, `dL/drho = dL/dp_C`);
- the rotation part `W = R^T` of the view matrix enters directly, with
  `W' = W Exp(-[theta]_x)` for the left perturbation above. This second term is
  added to `dL/dtheta` as `vee(dL/dW^T W - W^T dL/dW)`, where
  `dL/dW = (2 V J_o g) M^T` and `M = T * SQJ`.

### Per-pixel variance

For each pixel and channel `c`:

```text
M_c = sum_i w_i c_i + T_final bg_c
S_c = sum_i w_i (sigma2_i + c_i^2) + T_final bg_c^2
V_c = clamp(S_c - M_c^2, 1e-6, 0.2)
w_i = alpha_i T_i
```

The background is treated as a deterministic colour, so it contributes
`T_final * bg_c^2` to the second moment and no variance of its own. The backward
pass differentiates through both `S_c` and `M_c`, so gradients flow to the
per-splat variance (`dL/dsigma2_i = w_i dL/dV_c`), to colors
(`2 w_i (c_i - M_c) dL/dV_c`), and to the blending factors. The forward clamp is
honoured in the backward pass by zeroing gradients at the clamp bounds.

## Notes on the merge

- The omnidirectional forward covariance and projection are taken from the ODGS
  branch. The backward covariance chain is evaluated with the full
  spherical Jacobian `J_o = W * T * SQJ` (rather than the intermediate
  rotation matrix), which keeps the 3D-covariance gradients consistent with the
  forward model.
- The quaternion-based pose gradient of the omni branch is replaced by the SE(3)
  Lie-algebra Jacobian from the variance branch. Contributions from the 2D
  projection, the omnidirectional depth, and the spherical 2D covariance are
  all included; the Jacobian is evaluated at each Gaussian's camera-space
  centre `p_C = R m + t`. The covariance path additionally propagates through
  the rotation part `W` of the view matrix, so it remains correct even when the
  camera-space direction is held fixed during the pose perturbation.
  View-dependent SH colours depend on `campos`, which this function receives as
  an external tensor, so that path is left to the caller's pose graph.
- The variance backward is analytic and not a straight-through approximation.
- `lat`, `lon`, and `psi` are always returned for compatibility with
  omnidirectional densification strategies.

## Citation

If you use this code, please cite the parent works:

```bibtex
@inproceedings{tran2026varsplat,
  title   = {VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM},
  author  = {Tran, Anh Thuan and Kosecka, Jana},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year    = {2026}
}
```

```bibtex
@inproceedings{Matsuki:Murai:etal:CVPR2024,
  title     = {{G}aussian {S}platting {SLAM}},
  author    = {Hidenobu Matsuki and Riku Murai and Paul H. J. Kelly and Andrew J. Davison},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024}
}
```

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number  = {4},
  volume  = {42},
  year    = {2023}
}
```
