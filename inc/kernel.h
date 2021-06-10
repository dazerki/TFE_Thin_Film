#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void flux_x(float *u, int rho);

__global__ void flux_y(float *u, int rho);

#endif
