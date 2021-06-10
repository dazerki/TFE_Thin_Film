#include <cuda.h>
#include <stdio.h>

#include "kernel.h"

__global__ void flux_x(float *u, int rho)
{
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
  int rho_ij;
	int nx = N_DISCR;
	int ny = N_DISCR;
	int di = 1;
	int dj = 0;


	int i = (int) k % nx;
	int j = (int) k / nx;

	rho_ij = ((dj+1)*i + (di+1)*j + rho) % 4;

	if (rho_ij == 3){

		int i_p, j_p;
		float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
		float u_p, u_q;
	  float h = 1.0f/nx;

	  float tau = DELTA_T ;
		float e = EPSILON;
		float eta = ETA;
		float G = ZETA;
		if (i==0){
			i_p = nx - 1;
			j_p = j - dj;
		} else{
			i_p = i - di;
			j_p = j - dj;
		}

		if(i==0 || i==1 || i==nx-1 || j==0 || j==1 || j==ny-1){
			lap_q = (u[nx*((j+nx)%nx) + ((i+1+nx)%nx)] + u[nx*((j+1+nx)%nx) + ((i+nx)%nx)] + u[nx*((j-1+nx)%nx) + ((i+nx)%nx)]);
			lap_p = (u[nx*((j_p+nx)%nx) + (i_p-1+nx)%nx] + u[nx*((j_p+1+nx)%nx) + (i_p+nx)%nx] + u[nx*((j_p-1+nx)%nx) + (i_p+nx)%nx]);
		} else {
			lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(j-1) + i]);
			lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p+1) + i_p] + u[nx*(j_p-1) + i_p]);
		}


		u_p = u[nx*j_p + i_p];
		u_q = u[nx*j + i];

		W_q = G*(ny-j-0.5f)*h;
		W_p = G*(ny-j_p-0.5f)*h;

		M = 2.0f * u_p*u_p * u_q*u_q /(3.0f*(u_q + u_p));


		theta = h*h + (tau*M*(10.0f*e + 2.0f*eta ));
		f = (M*h/(theta)) * (eta*(u_p - u_q) + (e)*(lap_q - lap_p + 5.0f*(u_p-u_q)) + W_p-W_q);

		float val = tau*f/h;
		if(u_p<val){
			if(u_p > -u_q){
				delta_u = u_p;
			} else {
				delta_u = -u_q;
			}
		} else{
			if(val > -u_q){
				delta_u = val;
			} else {
				delta_u = -u_q;
			}
		}

		if(i == 0){
			u[nx*((j+nx)%nx) + ((i+nx)%nx)] += delta_u;
			u[nx*((j_p+nx)%nx) + ((i_p+nx)%nx)] -= delta_u;
		} else {
			u[nx*j + i] += delta_u;
			u[nx*j_p + i_p] -= delta_u;
		}


	}
}

__global__ void flux_y(float *u, int rho)
{
	//int k = blockIdx.x * blockDim.x + threadIdx.x;
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
  int rho_ij;
	int nx = N_DISCR;
	int ny = N_DISCR;
	int di = 0;
	int dj = 1;


	int i = (int) k % nx;
	int j = (int) k / nx;


	rho_ij = ((dj+1)*i + (di+1)*j + rho) % 4;
	if (rho_ij == 3){

		float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
		int i_p, j_p;

		float u_p, u_q;
	  float h = 1.0f/nx;

		float tau = DELTA_T ;
		float e = EPSILON;
		float eta = ETA;
		float G = ZETA;
		if (j==0){
			i_p = i - di;
			j_p = ny - 1;
		} else {
			i_p = i - di;
			j_p = j - dj;
		}


		if(i==0 || i==1 || i==nx-1 || j==0 || j==1 || j==ny-1){
			lap_q = (u[nx*((j+nx)%nx) + (i+1+nx)%nx] + u[nx*((j+1+nx)%nx) + (i+nx)%nx] + u[nx*((j+nx)%nx) + (i-1+nx)%nx]);
			lap_p = (u[nx*((j_p+nx)%nx) + (i_p-1+nx)%nx] + u[nx*((j_p+nx)%nx) + (i_p+1+nx)%nx] + u[nx*((j_p-1+nx)%nx) + (i_p+nx)%nx]);
		} else {
			lap_q = (u[nx*j + (i+1)] + u[nx*(j+1) + i] + u[nx*(j) + i-1]);
			lap_p = (u[nx*j_p + (i_p-1)] + u[nx*(j_p) + i_p+1] + u[nx*(j_p-1) + i_p]);
		}

		u_p = u[nx*j_p + i_p];
		u_q = u[nx*j + i];

		W_q = G*(ny-j-0.5f)*h;

		if(j==0){
			W_p = G*(ny-(-1.0f)-0.5f)*h;
		}else{
			W_p = G*(ny-j_p-0.5f)*h;
		}

		M = 2.0f * u_q*u_q * u_p*u_p /(3.0f*(u_q + u_p));

		theta = h*h + (tau*M*(10.0f*e + 2.0f*eta));
		f = (M*h/(theta)) * (eta*(u_p - u_q) + (e)*(lap_q - lap_p + 5.0f*(u_p-u_q)) + W_p-W_q);

		float val = tau*f/h;
		if(u_p<val){
			if(u_p > -u_q){
				delta_u = u_p;
			} else {
				delta_u = -u_q;
			}
		} else{
			if(val > -u_q){
				delta_u = val;
			} else {
				delta_u = -u_q;
			}
		}

		if(j == 0){
			u[nx*((j+nx)%nx) + ((i+nx)%nx)] += delta_u;
			u[nx*((j_p+nx)%nx) + ((i_p+nx)%nx)] -= delta_u;
		} else {
			u[nx*j + i] += delta_u;
			u[nx*j_p + i_p] -= delta_u;
		}
  }
}
