# Differential Gaussian Rasterization with Camera Pose Jacobians and Per-Splat Variance

Modified rasterization engine used in [VarSplat](https://github.com/anhthuan1999/varsplat). Built on top of [diff-gaussian-rasterization-w-pose](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose) from "Gaussian Splatting SLAM", this fork adds:

* Uncertainty map rendering via per-splat appearance variance.
* Analytical gradient for SE(3) camera poses.
* Analytical gradient for rendered depth.

## Install

If you can make use of it in your own research, please be so kind to cite all three papers.

## Citation

If you use this code, please cite:

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
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
