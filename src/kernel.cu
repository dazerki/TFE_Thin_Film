#include <cuda.h>
#include <stdio.h>

#include "kernel.h"

__global__ void flux_x(float *u, int rho)
{

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
		float H_E, k_E;
		float H_p, H_q;
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

		if((j-2)%102 == 0 || (j-1)%102 == 0 || (j)%102 == 0){
			H_q = 0.03f;
			H_E = 0.03f;
			k_E = 0.06f;

		} else if((i-2)%102 == 0 || (i-1)%102 == 0 || (i)%102 == 0){
			if((i+j-3)%204 > 102 || (i+j-2)%204 > 102 || (i+j-1)%204 > 102){
				H_q = 0.03f;
				H_E = 0.0f;
				k_E = 0.0f;
			} else {
				H_q = 0.0f;
				H_E = 0.0f;
				k_E = 0.0f;
			}
		} else{
			H_q = 0.0f;
			H_E = 0.0f;
			k_E = 0.0f;
		}

		if((j_p-2)%102 == 0 || (j_p-1)%102 == 0 || (j_p)%102 ==  0){
			H_p = 0.03f;
		} else if((i_p-2)%102 == 0 || (i_p-1)%102 == 0 || (i_p)%102 == 0){
			if((i_p+j_p-3)%204 > 102 || (i_p+j_p-2)%204 > 102 || (i_p+j_p-1)%204 > 102){
				H_q = 0.03f;
			} else {
				H_q = 0.0f;
			}
		} else {
			H_p = 0.0f;
		}

		W_q = G*(ny-j-0.5f)*h - H_q;
		W_p = G*(ny-j_p-0.5f)*h - H_p;

		M = 2.0f * u_p*u_p * u_q*u_q /(3.0f*(u_q + u_p)) + (e/6.0f)*u_q*u_q*u_p*u_p*(H_E+k_E);

		theta = h*h + (tau*M*(10.0f*e + 2.0f*eta));
		f = -(M*h/(theta)) * ((5.0f*e + eta)*(u_q - u_p) - e*(lap_q - lap_p) + W_q-W_p);

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

		u[nx*j + i] = u_q + delta_u;
		u[nx*j_p + i_p] = u_p - delta_u;

	}
}

__global__ void flux_y(float *u, int rho)
{
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
		float H_p, H_q, H_E, k_E;
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

		if((j-2)%102 == 0 || (j-1)%102 == 0 || (j)%102 == 0){
			H_q = 0.03f;
			H_E = 0.0f;
			k_E = 0.0f;
		} else if((i-2)%102 == 0 || (i-1)%102 == 0 || (i)%102 == 0){
			if((i+j-3)%204 > 102 || (i+j-2)%204 > 102 || (i+j-1)%204 > 102){
				H_q = 0.03f;
				H_E = 0.06f;
				k_E = 0.06f;
			} else {
				H_q = 0.0f;
				H_E = 0.0f;
				k_E = 0.0f;
			}
		} else{
			H_q = 0.0f;
			H_E = 0.0f;
			k_E = 0.0f;
		}

		if((j_p-2)%102 == 0 || (j_p-1)%102 == 0 || (j_p)%102 ==  0){
			H_p = 0.03f;
		} else if((i_p-2)%102 == 0 || (i_p-1)%102 == 0 || (i_p)%102 == 0){
			if((i_p+j_p-3)%204 > 102 || (i_p+j_p-2)%204 > 102 || (i_p+j_p-1)%204 > 102){
				H_q = 0.03f;
			} else {
				H_q = 0.0f;
			}
		} else {
			H_p = 0.0f;
		}

		W_q = G*(ny-j-0.5f)*h - H_q;

		if(j==0){
			W_p = G*(ny-(-1.0f)-0.5f)*h - H_p;
		}else{
			W_p = G*(ny-j_p-0.5f)*h - H_p;
		}

		M = 2.0f * u_p*u_p * u_q*u_q /(3.0f*(u_q + u_p)) + (e/6.0f)*u_q*u_q*u_p*u_p*(H_E+k_E);

		theta = h*h + (tau*M*(10.0f*e + 2.0f*eta));
		f = -(M*h/(theta)) * ((5.0f*e + eta)*(u_q - u_p) - e*(lap_q - lap_p) + W_q-W_p);

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

		u[nx*j + i] = u_q + delta_u;
		u[nx*j_p + i_p] = u_p - delta_u;
  }
}
