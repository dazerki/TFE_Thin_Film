#include <cuda.h>
#include <stdio.h>

#include "kernel.h"

__global__ void flux_x(float *u, float *data_3D, float *data_edge, int rho)
{
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
  int rho_ij;
	int nx = 512;
	int ny = 512;
	int di = 1;
	int dj = 0;


	int i = (int) k % nx;
	int j = (int) k / nx;

	rho_ij = ((dj+1)*i + (di+1)*j + rho) % 4;

	if (rho_ij == 3){

		int i_p, j_p;
		float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
		float H_p, H_q, T_p, T_q, ct_p, ct_q;
		float k_E, H_E;
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

		H_p = data_3D[(nx*j_p + i_p)*3];
		H_q = data_3D[(nx*j + i)*3];

		T_p = data_3D[(nx*j_p + i_p)*3 + 1];
		T_q = data_3D[(nx*j + i)*3 + 1];

		ct_p = data_3D[(nx*j_p + i_p)*3 + 2];
		ct_q = data_3D[(nx*j + i)*3 + 2];

		if(((nx+1)*j + i)*2 + 1 > (nx+1)*nx*2){
			printf("i = %d, j = %d \n", i, j);
		}
		k_E = data_edge[((nx+1)*j + i)*2];
		H_E = data_edge[((nx+1)*j + i)*2 + 1];

		W_q = G*(ny-j-0.5f)*h - H_q;
		W_p = G*(ny-j_p-0.5f)*h - H_p;

		M = 2.0f * u_p*u_p * u_q*u_q /(3.0f*(u_q + u_p)) + (e/6.0f)*u_q*u_q*u_p*u_p*(H_E+k_E);


		theta = h*h + (tau*M*(5.0f*e + 2.0f*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
		f = (M*h/(theta)) * (eta*(u_p - u_q) + (e)*(lap_q - lap_p + 5.0f*(u_p-u_q)) + W_p-W_q + e*((G*ct_q - T_q)*u_q - (G*ct_p - T_p)*u_p));

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

__global__ void flux_y(float *u, float *data_3D, float *data_edge, int rho)
{
	//int k = blockIdx.x * blockDim.x + threadIdx.x;
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
  int rho_ij;
	int nx = 512;
	int ny = 512;
	int di = 0;
	int dj = 1;


	int i = (int) k % nx;
	int j = (int) k / nx;


	rho_ij = ((dj+1)*i + (di+1)*j + rho) % 4;
	if (rho_ij == 3){

		float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
		float H_p, H_q, T_p, T_q, ct_p, ct_q;
		float k_E, H_E;
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

		H_p = data_3D[(nx*j_p + i_p)*3];
		H_q = data_3D[(nx*j + i)*3];

		T_p = data_3D[(nx*j_p + i_p)*3 + 1];
		T_q = data_3D[(nx*j + i)*3 + 1];

		ct_p = data_3D[(nx*j_p + i_p)*3 + 2];
		ct_q = data_3D[(nx*j + i)*3 + 2];

		k_E = data_edge[((nx)*j + i)*2];
		H_E = data_edge[((nx)*j + i)*2 + 1];

		W_q = G*(ny-j-0.5f)*h - H_q;

		if(j==0){
			W_p = G*(ny-(-1.0f)-0.5f)*h - H_p;
		}else{
			W_p = G*(ny-j_p-0.5f)*h - H_p;
		}

		M = 2.0f * u_q*u_q * u_p*u_p /(3.0f*(u_q + u_p)) + (e/6.0f)*u_q*u_q*u_p*u_p*(H_E+k_E);

		theta = h*h + (tau*M*(5.0f*e + 2.0f*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
		f = (M*h/(theta)) * (eta*(u_p - u_q) + (e)*(lap_q - lap_p + 5.0f*(u_p-u_q)) + W_p-W_q + e*((G*ct_q - T_q)*u_q - (G*ct_p - T_p)*u_p));

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
