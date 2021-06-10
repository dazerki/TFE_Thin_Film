#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void update_u(float *u, float* flux_x, float* flux_y, int nx);

__global__ void flux_block(float *u, float* data_3D_gpu, float* data_edge_gpu, float* flx_x, float* flx_y, int nx);

#endif
