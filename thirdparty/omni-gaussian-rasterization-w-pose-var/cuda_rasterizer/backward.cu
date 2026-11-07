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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Add an SE(3) pose Jacobian contribution for one Gaussian.
// dL_dp is the gradient w.r.t. the Gaussian centre p_C in camera coordinates.
// The left-perturbed pose is T_CW <- Exp([rho, theta]) T_CW, hence
// p_C = R m + t, dp_C/drho = I and dp_C/dtheta = -[p_C]_x.
// Therefore dL/dtheta = [p_C]_x dL/dp = p_C x dL/dp.
__device__ __forceinline__ void accumulatePoseGradient(float* dL_dtau, int idx, const float3& dL_dp, const float3& p_cam)
{
	const float3 dL_dtheta = {
		p_cam.y * dL_dp.z - p_cam.z * dL_dp.y,
		p_cam.z * dL_dp.x - p_cam.x * dL_dp.z,
		p_cam.x * dL_dp.y - p_cam.y * dL_dp.x };

	dL_dtau[6 * idx + 0] += dL_dp.x;
	dL_dtau[6 * idx + 1] += dL_dp.y;
	dL_dtau[6 * idx + 2] += dL_dp.z;
	dL_dtau[6 * idx + 3] += dL_dtheta.x;
	dL_dtau[6 * idx + 4] += dL_dtheta.y;
	dL_dtau[6 * idx + 5] += dL_dtheta.z;
}

// Backward version of INVERSE 2D covariance matrix computation in the
// omnidirectional (equirectangular) image.
__global__ void computeOmniCov2D(int P,
	const int height,
	const int width,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov,
	float* dL_dtau,
	bool mapping_mode,
	bool tracking_mode)
{
	// The derivation of the backward steps is outlined by equations (11)-(12)
	// of Appendix A.3 in "ODGS" (Suyoung Lee et al., 2024).
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;
	if (!mapping_mode && !tracking_mode)
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant
	// intermediate forward results needed in the backward.
	const float e = 0.0000001f;
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, viewmatrix);

	float dist = sqrt(t.x * t.x + t.y * t.y + t.z * t.z) + e;
	float dist_xz = sqrt(t.x * t.x + t.z * t.z) + e;
	float lat = atan2(-t.y, dist_xz);
	float lon = atan2(t.x, t.z);

	const float x_scale = width / (2 * M_PI);
	const float y_scale = height / (M_PI);

	glm::mat3 SQJ = glm::mat3(
		x_scale / ((cos(lat) + e) * dist), 0, 0,
		0, y_scale / dist, 0,
		0, 0, 0);

	glm::mat3 T = glm::mat3(
		cos(lon), 0.0f, -sin(lon),
		sin(lat) * sin(lon), cos(lat), sin(lat) * cos(lon),
		cos(lat) * sin(lon), -sin(lat), cos(lat) * cos(lon));

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 J_o = W * T * SQJ;

	glm::mat3 cov2D = glm::transpose(J_o) * glm::transpose(Vrk) * J_o;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + e);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		dL_dcov[6 * idx + 0] = (J_o[0][0] * J_o[0][0] * dL_da + J_o[0][0] * J_o[1][0] * dL_db + J_o[1][0] * J_o[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (J_o[0][1] * J_o[0][1] * dL_da + J_o[0][1] * J_o[1][1] * dL_db + J_o[1][1] * J_o[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (J_o[0][2] * J_o[0][2] * dL_da + J_o[0][2] * J_o[1][2] * dL_db + J_o[1][2] * J_o[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		dL_dcov[6 * idx + 1] = 2 * J_o[0][0] * J_o[0][1] * dL_da + (J_o[0][0] * J_o[1][1] + J_o[0][1] * J_o[1][0]) * dL_db + 2 * J_o[1][0] * J_o[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * J_o[0][0] * J_o[0][2] * dL_da + (J_o[0][0] * J_o[1][2] + J_o[0][2] * J_o[1][0]) * dL_db + 2 * J_o[1][0] * J_o[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * J_o[0][2] * J_o[0][1] * dL_da + (J_o[0][1] * J_o[1][2] + J_o[0][2] * J_o[1][1]) * dL_db + 2 * J_o[1][1] * J_o[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix J_o
	float dL_dT00 = 2 * (J_o[0][0] * Vrk[0][0] + J_o[0][1] * Vrk[0][1] + J_o[0][2] * Vrk[0][2]) * dL_da +
		(J_o[1][0] * Vrk[0][0] + J_o[1][1] * Vrk[0][1] + J_o[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (J_o[0][0] * Vrk[1][0] + J_o[0][1] * Vrk[1][1] + J_o[0][2] * Vrk[1][2]) * dL_da +
		(J_o[1][0] * Vrk[1][0] + J_o[1][1] * Vrk[1][1] + J_o[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (J_o[0][0] * Vrk[2][0] + J_o[0][1] * Vrk[2][1] + J_o[0][2] * Vrk[2][2]) * dL_da +
		(J_o[1][0] * Vrk[2][0] + J_o[1][1] * Vrk[2][1] + J_o[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (J_o[1][0] * Vrk[0][0] + J_o[1][1] * Vrk[0][1] + J_o[1][2] * Vrk[0][2]) * dL_dc +
		(J_o[0][0] * Vrk[0][0] + J_o[0][1] * Vrk[0][1] + J_o[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (J_o[1][0] * Vrk[1][0] + J_o[1][1] * Vrk[1][1] + J_o[1][2] * Vrk[1][2]) * dL_dc +
		(J_o[0][0] * Vrk[1][0] + J_o[0][1] * Vrk[1][1] + J_o[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (J_o[1][0] * Vrk[2][0] + J_o[1][1] * Vrk[2][1] + J_o[1][2] * Vrk[2][2]) * dL_dc +
		(J_o[0][0] * Vrk[2][0] + J_o[0][1] * Vrk[2][1] + J_o[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ10 = W[0][0] * dL_dT10 + W[0][1] * dL_dT11 + W[0][2] * dL_dT12;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	// Gradients of loss w.r.t. the non-zero entries of M = T * SQJ.
	// These follow the math convention M[row][col] (glm's M[col][row]).
	const float dL_dM00 = dL_dJ00;
	const float dL_dM01 = dL_dJ10;
	const float dL_dM20 = dL_dJ02;
	const float dL_dM11 = dL_dJ11;
	const float dL_dM21 = dL_dJ12;

	// Scales used by SQJ and their derivative with respect to lat.
	const float sx = x_scale / ((cos(lat) + e) * dist);
	const float sy = y_scale / dist;
	const float dsx_dlat = sx * sin(lat) / (cos(lat) + e);

	// Non-zero entries of M (math indexing).
	const float sin_lat = sin(lat);
	const float cos_lat = cos(lat);
	const float sin_lon = sin(lon);
	const float cos_lon = cos(lon);
	const float M00 = sx * cos_lon;
	const float M01 = sy * sin_lat * sin_lon;
	const float M11 = sy * cos_lat;
	const float M20 = -sx * sin_lon;
	const float M21 = sy * sin_lat * cos_lon;

	// Gradients of loss w.r.t. lat, lon and dist through M = T * SQJ.
	const float dL_dlat =
		dL_dM00 * (dsx_dlat * cos_lon) +
		dL_dM01 * (sy * cos_lat * sin_lon) +
		dL_dM11 * (-sy * sin_lat) +
		dL_dM20 * (-dsx_dlat * sin_lon) +
		dL_dM21 * (sy * cos_lat * cos_lon);

	const float dL_dlon =
		dL_dM00 * (-sx * sin_lon) +
		dL_dM01 * (sy * sin_lat * cos_lon) +
		dL_dM20 * (-sx * cos_lon) +
		dL_dM21 * (-sy * sin_lat * sin_lon);

	const float dL_ddist = -(dL_dM00 * M00 + dL_dM01 * M01 +
		dL_dM11 * M11 + dL_dM20 * M20 + dL_dM21 * M21) / dist;

	// lat = atan2(-ty, dist_xz), lon = atan2(tx, tz), dist = |t|.
	const float rx = sqrt(t.x * t.x + t.z * t.z);
	const float rr = rx + e;
	const float qlat = rr * rr + t.y * t.y;
	const float lon_denom = 1.0f / (t.x * t.x + t.z * t.z + e);
	const float inv_rx = 1.0f / (rx + e);

	const float dlat_dtx = (t.y / qlat) * t.x * inv_rx;
	const float dlat_dty = -rr / qlat;
	const float dlat_dtz = (t.y / qlat) * t.z * inv_rx;

	const float dlon_dtx = t.z * lon_denom;
	const float dlon_dtz = -t.x * lon_denom;

	const float inv_tnorm = 1.0f / (sqrt(t.x * t.x + t.y * t.y + t.z * t.z) + e);

	// Position in camera space: dL/dt.
	const float dL_dtx = dL_dlat * dlat_dtx + dL_dlon * dlon_dtx + dL_ddist * t.x * inv_tnorm;
	const float dL_dty = dL_dlat * dlat_dty + dL_ddist * t.y * inv_tnorm;
	const float dL_dtz = dL_dlat * dlat_dtz + dL_dlon * dlon_dtz + dL_ddist * t.z * inv_tnorm;

	if (mapping_mode)
	{
		// Account for transformation of mean to t.
		// t = transformPoint4x3(mean, viewmatrix);
		float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, viewmatrix);

		// Gradients of loss w.r.t. Gaussian means, but only the portion
		// that is caused because the mean affects the covariance matrix.
		// Additional mean gradient is accumulated in BACKWARD::preprocess.
		dL_dmeans[idx] = dL_dmean;
	}

	if (tracking_mode)
	{
		// t is the Gaussian centre in camera coordinates p_C; the left
		// perturbation moves it as dp_C/drho = I, dp_C/dtheta = -[p_C]_x.
		accumulatePoseGradient(dL_dtau, idx, make_float3(dL_dtx, dL_dty, dL_dtz), t);

		// Additional pose dependence through the rotation part W of the view
		// matrix. Since J_o = W * M with M = T * SQJ (holding the camera-space
		// direction fixed here), we have dL/dJ = 2 V J g and dL/dW = dL/dJ M^T.
		// With W' = W Exp(-[theta]_x) this yields
		// dL/dtheta = vee(dL/dW^T W - W^T dL/dW).
		const glm::mat3 g = glm::mat3(
			dL_da, 0.5f * dL_db, 0.0f,
			0.5f * dL_db, dL_dc, 0.0f,
			0.0f, 0.0f, 0.0f);
		const glm::mat3 M = T * SQJ;
		const glm::mat3 dL_dJ = 2.0f * Vrk * J_o * g;
		const glm::mat3 dL_dW = dL_dJ * glm::transpose(M);
		const glm::mat3 A = glm::transpose(dL_dW) * W;

		dL_dtau[6 * idx + 3] += A[1][2] - A[2][1];
		dL_dtau[6 * idx + 4] += A[2][0] - A[0][2];
		dL_dtau[6 * idx + 5] += A[0][1] - A[1][0];
	}
}

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;
	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const int W, int H,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dcov3D,
	const float* dL_ddepth,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dtau,
	bool mapping_mode,
	bool tracking_mode)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 mean = means[idx];
	float3 t = transformPoint4x3(mean, viewmatrix);

	float dist = sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
	float inv_dist = 1.0f / (dist + 0.0000001f);
	float inv_dist2 = 1.0f / (dist * dist + 0.0000001f);
	float dist_xz = sqrt(t.x * t.x + t.z * t.z);
	float inv_dist_xz = 1.0f / (dist_xz + 0.0000001f);
	float dist_xz2 = dist_xz * dist_xz;

	const float x_scale = W / (2 * M_PI) / (dist_xz2 + 0.0000001f);
	const float y_scale = H / M_PI * inv_dist2 * inv_dist_xz;

	if (mapping_mode || tracking_mode)
	{
		// Camera-space gradient of the projected 2D mean.
		float3 dL_dp;
		dL_dp.x = x_scale * t.z * dL_dmean2D[idx].x - y_scale * t.x * t.y * dL_dmean2D[idx].y;
		dL_dp.y = (H / M_PI) * inv_dist2 * dist_xz * dL_dmean2D[idx].y;
		dL_dp.z = -x_scale * t.x * dL_dmean2D[idx].x - y_scale * t.y * t.z * dL_dmean2D[idx].y;

		// Camera-space gradient of the omnidirectional depth
		// (= distance from the camera centre).
		dL_dp.x += t.x * inv_dist * dL_ddepth[idx];
		dL_dp.y += t.y * inv_dist * dL_ddepth[idx];
		dL_dp.z += t.z * inv_dist * dL_ddepth[idx];

		if (mapping_mode)
		{
			// Map the camera-space gradient back to world space. The
			// viewmatrix stores a column-major world-to-camera transform,
			// so this applies the transpose of its rotation part (R^T).
			glm::vec3 dL_dmean = transformVec4x3Transpose(
				make_float3(dL_dp.x, dL_dp.y, dL_dp.z), viewmatrix);

			// Add the part caused by the projection and the depth, on top
			// of the covariance contribution written by computeOmniCov2D.
			dL_dmeans[idx] += dL_dmean;

			// Compute gradient updates due to computing colors from SHs
			if (shs)
				computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

			// Compute gradient updates due to computing covariance from scale/rotation
			if (scales)
				computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
		}

		if (tracking_mode)
		{
			// t is the Gaussian centre in camera coordinates p_C.
			accumulatePoseGradient(dL_dtau, idx, dL_dp, t);
		}
	}
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_depth,
	const float* __restrict__ final_color,
	const float* __restrict__ final_variance,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths,
	const float* __restrict__ color_variance,
	const float* __restrict__ dL_dvariance,
	float* __restrict__ dL_dcolor_variance)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpix[C];
	float accum_rec_depth = 0;
	float dL_ddepth;

	// Optional per-splat-variance loss. final_color is the rendered mean
	// and final_variance is the clamped forward variance (used to honour
	// the clamp in the backward pass).
	const bool has_variance = inside && dL_dvariance != nullptr && final_color != nullptr;
	float dL_dV[C] = { 0 };
	float accum_rec_q[C] = { 0 };
	float last_q[C] = { 0 };

	if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpix[i] = dL_dpixels[i * H * W + pix_id];
		dL_ddepth = dL_dpixels_depth[pix_id];

		if (has_variance) {
			for (int i = 0; i < C; i++) {
				float g = dL_dvariance[i * H * W + pix_id];
				if (final_variance != nullptr) {
					float v = final_variance[i * H * W + pix_id];
					// The forward pass writes the exact clamp bounds.
					if (v <= 1e-6f || v >= 0.2f)
						g = 0.0f;
				}
				dL_dV[i] = g;
			}
		}
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport coordinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in reverse order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depths[block.thread_rank()] = depths[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel pair).
			float dL_dalpha = 0.0f;
			float dL_dalpha_var_noT = 0.0f;
			float dL_dalpha_var_bg = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpix[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);

				if (has_variance)
				{
					const float var_ch = (color_variance != nullptr)
						? fmaxf(color_variance[global_id * 3 + ch], 0.0f)
						: 0.0f;
					const float q = c * c + var_ch;

					// The second moment S = sum_i w_i q_i is blended like
					// a colour image, with "pixel gradient" dL/dV.
					accum_rec_q[ch] = last_alpha * last_q[ch] + (1.f - last_alpha) * accum_rec_q[ch];
					last_q[ch] = q;
					dL_dalpha_var_noT += (q - accum_rec_q[ch]) * dL_dV[ch];

					// M = sum_i w_i c_i + T_final * bg. Its contribution
					// carries the "pixel gradient" -2 M dL/dV.
					const float h = -2.0f * final_color[ch * H * W + pix_id] * dL_dV[ch];
					dL_dalpha_var_noT += (c - accum_rec[ch]) * h;
					// Background: M contributes bg*dL/dM and the
					// deterministic second moment T*bg^2 contributes
					// bg^2*dL/dV through the same T_final factor.
					dL_dalpha_var_bg += bg_color[ch] * h + bg_color[ch] * bg_color[ch] * dL_dV[ch];

					// dV/dc_i = 2 w_i (c_i - M), with w_i = alpha_i T_i.
					atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * (2.0f * c * dL_dV[ch] + h));

					// dV/dvar_i = w_i.
					if (dL_dcolor_variance != nullptr)
						atomicAdd(&(dL_dcolor_variance[global_id * 3 + ch]), dchannel_dcolor * dL_dV[ch]);
				}
			}

			const float depth = collected_depths[j];
			accum_rec_depth = last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
			last_depth = depth;

			// Gradient w.r.t. alpha a_i => (d_i - accum_rec_depth) * dL/dD
			dL_dalpha += (depth - accum_rec_depth) * dL_ddepth;

			// Gradient w.r.t. depth d_i => w_i * dL/dD
			atomicAdd(&(dL_ddepths[global_id]), dchannel_dcolor * dL_ddepth);

			dL_dalpha *= T;
			if (has_variance)
				dL_dalpha += T * dL_dalpha_var_noT;

			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpix[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * (bg_dot_dpixel + (has_variance ? dL_dalpha_var_bg : 0.0f));

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const int width, int height,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_ddepth,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dtau,
	bool mapping_mode,
	bool tracking_mode)
{
	// Propagate gradients for the path of 2D conic matrix computation.
	// Somewhat long, thus it is its own kernel rather than being part of
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.
	if (mapping_mode || tracking_mode)
	{
		computeOmniCov2D << <(P + 255) / 256, 256 >> > (
			P, height, width,
			means3D,
			radii,
			cov3Ds,
			viewmatrix,
			dL_dconic,
			(float3*)dL_dmean3D,
			dL_dcov3D,
			dL_dtau,
			mapping_mode,
			tracking_mode);
	}

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desired), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		width, height,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		campos,
		(float3*)dL_dmean2D,
		dL_dcov3D,
		dL_ddepth,
		(float3*)dL_dmean3D,
		dL_dcolor,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dtau,
		mapping_mode,
		tracking_mode);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixels_depth,
	const float* final_color,
	const float* final_variance,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths,
	const float* color_variance,
	const float* dL_dvariance,
	float* dL_dcolor_variance)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dpixels_depth,
		final_color,
		final_variance,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths,
		color_variance,
		dL_dvariance,
		dL_dcolor_variance);
}
