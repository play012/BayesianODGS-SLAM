/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
#include <functional>
#include "rasterize_points.h"
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& color_variance,
	const bool return_variance)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({ NUM_CHANNELS, H, W }, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({ H, W }, 0.0, float_opts);
	torch::Tensor out_opacity = torch::full({ H, W }, 0.0, float_opts);
	torch::Tensor radii = torch::full({ P }, 0, int_opts);
	torch::Tensor psi = torch::full({ P }, 0, float_opts);
	torch::Tensor lat = torch::full({ P }, 0, float_opts);
	torch::Tensor lon = torch::full({ P }, 0, float_opts);
	torch::Tensor n_touched = torch::full({ P }, 0, int_opts);

	// Allocate 3-channel variance output tensor if requested.
	torch::Tensor out_variance = torch::empty({ 0 }, float_opts);
	if (return_variance) {
		out_variance = torch::full({ NUM_CHANNELS, H, W }, 0.0, float_opts);
	}

	torch::TensorOptions options = means3D.options().dtype(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({ 0 }, options);
	torch::Tensor binningBuffer = torch::empty({ 0 }, options);
	torch::Tensor imgBuffer = torch::empty({ 0 }, options);
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	// Keep a contiguous copy of the per-splat variance alive for the launch.
	torch::Tensor color_variance_contig;
	const float* variance_ptr = nullptr;
	if (return_variance && color_variance.defined() && color_variance.numel() > 0) {
		TORCH_CHECK(color_variance.dim() == 2 && color_variance.size(0) == P && color_variance.size(1) == NUM_CHANNELS,
			"color_variance must have shape (num_points, NUM_CHANNELS)");
		TORCH_CHECK(color_variance.is_cuda(), "color_variance must be a CUDA tensor");
		color_variance_contig = color_variance.contiguous();
		variance_ptr = color_variance_contig.data_ptr<float>();
	}
	float* out_variance_ptr = return_variance ? out_variance.data_ptr<float>() : nullptr;

	int rendered = 0;
	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0) {
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data_ptr<float>(),
			W, H,
			means3D.contiguous().data_ptr<float>(),
			sh.numel() > 0 ? sh.contiguous().data_ptr<float>() : nullptr,
			colors.numel() > 0 ? colors.contiguous().data_ptr<float>() : nullptr,
			opacity.contiguous().data_ptr<float>(),
			scales.numel() > 0 ? scales.contiguous().data_ptr<float>() : nullptr,
			scale_modifier,
			rotations.numel() > 0 ? rotations.contiguous().data_ptr<float>() : nullptr,
			cov3D_precomp.numel() > 0 ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr,
			viewmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			prefiltered,
			out_color.contiguous().data_ptr<float>(),
			out_depth.contiguous().data_ptr<float>(),
			out_opacity.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			psi.contiguous().data_ptr<float>(),
			lat.contiguous().data_ptr<float>(),
			lon.contiguous().data_ptr<float>(),
			n_touched.contiguous().data_ptr<int>(),
			debug,
			variance_ptr,
			out_variance_ptr);
	}

	return std::make_tuple(rendered, out_color, out_depth, out_opacity, radii, psi, lat, lon, n_touched, out_variance, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const torch::Tensor& color_variance,
	const torch::Tensor& dL_dvariance,
	const torch::Tensor& final_color,
	const torch::Tensor& final_variance,
	const bool mapping_mode,
	const bool tracking_mode)
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if (sh.size(0) != 0) {
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans2D = torch::zeros({ P, 3 }, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({ P, NUM_CHANNELS }, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({ P, 1 }, means3D.options());
	torch::Tensor dL_dmeans3D = torch::zeros({ P, 3 }, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({ P, 6 }, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({ P, M, 3 }, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({ P, 3 }, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({ P, 4 }, means3D.options());
	torch::Tensor dL_dtau = torch::zeros({ P, 6 }, means3D.options());
	torch::Tensor dL_dcolor_variance = torch::zeros({ P, 3 }, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({ P, 2, 2 }, means3D.options());
	torch::Tensor dL_ddepth = torch::zeros({ P, 1 }, means3D.options());

	// Keep contiguous copies alive for the asynchronous kernel launches.
	torch::Tensor color_variance_contig;
	torch::Tensor dL_dvariance_contig;
	torch::Tensor final_color_contig;
	torch::Tensor final_variance_contig;

	const float* color_variance_ptr = nullptr;
	if (color_variance.defined() && color_variance.numel() > 0) {
		color_variance_contig = color_variance.contiguous();
		color_variance_ptr = color_variance_contig.data_ptr<float>();
	}
	const float* dL_dvariance_ptr = nullptr;
	if (dL_dvariance.defined() && dL_dvariance.numel() > 0) {
		dL_dvariance_contig = dL_dvariance.contiguous();
		dL_dvariance_ptr = dL_dvariance_contig.data_ptr<float>();
	}
	const float* final_color_ptr = nullptr;
	if (final_color.defined() && final_color.numel() > 0) {
		final_color_contig = final_color.contiguous();
		final_color_ptr = final_color_contig.data_ptr<float>();
	}
	const float* final_variance_ptr = nullptr;
	if (final_variance.defined() && final_variance.numel() > 0) {
		final_variance_contig = final_variance.contiguous();
		final_variance_ptr = final_variance_contig.data_ptr<float>();
	}

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::backward(P, degree, M, R,
			background.contiguous().data_ptr<float>(),
			W, H,
			means3D.contiguous().data_ptr<float>(),
			sh.numel() > 0 ? sh.contiguous().data_ptr<float>() : nullptr,
			colors.numel() > 0 ? colors.contiguous().data_ptr<float>() : nullptr,
			scales.numel() > 0 ? scales.contiguous().data_ptr<float>() : nullptr,
			scale_modifier,
			rotations.numel() > 0 ? rotations.contiguous().data_ptr<float>() : nullptr,
			cov3D_precomp.numel() > 0 ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr,
			viewmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data_ptr<float>(),
			dL_dout_depth.contiguous().data_ptr<float>(),
			final_color_ptr,
			final_variance_ptr,
			dL_dmeans2D.contiguous().data_ptr<float>(),
			dL_dconic.contiguous().data_ptr<float>(),
			dL_dopacity.contiguous().data_ptr<float>(),
			dL_dcolors.contiguous().data_ptr<float>(),
			dL_ddepth.contiguous().data_ptr<float>(),
			dL_dmeans3D.contiguous().data_ptr<float>(),
			dL_dcov3D.contiguous().data_ptr<float>(),
			dL_dsh.contiguous().data_ptr<float>(),
			dL_dscales.contiguous().data_ptr<float>(),
			dL_drotations.contiguous().data_ptr<float>(),
			dL_dtau.contiguous().data_ptr<float>(),
			mapping_mode,
			tracking_mode,
			debug,
			color_variance_ptr,
			dL_dvariance_ptr,
			dL_dcolor_variance.contiguous().data_ptr<float>());
	}

	return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dtau, dL_dcolor_variance);
}

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix)
{
	const int P = means3D.size(0);
	torch::Tensor present = torch::full({ P }, false, means3D.options().dtype(torch::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(P,
			means3D.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			present.contiguous().data_ptr<bool>());
	}

	return present;
}
