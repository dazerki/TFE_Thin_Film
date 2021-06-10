#include <cuda.h>
#include <stdio.h>

#include "kernel.h"

__global__ void flux_block(float *u, float* data_3D_gpu, float* data_edge_gpu, float* flx_x, float* flx_y, int nx){
	//position in the grid
	int pos_x = (blockIdx.x * blockDim.x + threadIdx.x)-2;
	int pos_y = (blockIdx.y * blockDim.y + threadIdx.y)-2;

	int pos_data3D_x = (blockIdx.x * blockDim.x + threadIdx.x)-1;
	int pos_data3D_y = (blockIdx.y * blockDim.y + threadIdx.y)-1;



	int pos_x_data = (blockIdx.x * blockDim.x + threadIdx.x);
	int pos_y_data = (blockIdx.y * blockDim.y + threadIdx.y);

	int pos_block_i = threadIdx.x  + 2;
	int pos_block_j = threadIdx.y  + 2;

	int pos_data3D_i = threadIdx.x+1;
	int pos_data3D_j = threadIdx.y+1;

	int pos_block_x = threadIdx.x;
	int pos_block_y = threadIdx.y;

	int size_u_line = (blockDim.x+3);
	int size_data_line = blockDim.x+1;
	int size_edge_line = blockDim.x;

	//access memory
	__shared__ float u_local[1225]; //4*dimX+3) * (4*dimY+3
	__shared__ float data_3D_local[3267]; // 3*(4*dimX+1)*(dimY*4+1)
	__shared__ float data_edge_local[2178]; // 2*(4*dimX+1)*(dimY*4+1)
	//u local
	if(threadIdx.x == 0){
		if(threadIdx.y == 0){
			u_local[size_u_line+1] = u[((pos_y+1+nx)%nx)*nx + ((pos_x+1+nx)%nx)];
		} else if(threadIdx.y == blockDim.y-1){
			u_local[size_u_line*(blockDim.x+2)+1] = u[((pos_y+3+nx)%nx)*nx + ((pos_x+1+nx)%nx)];
		}
		for(int i=0; i<2; i++){
			u_local[size_u_line*(pos_block_y+2) + (i+pos_block_x)] = u[((pos_y+2+nx)%nx)*nx + ((pos_x+i+nx)%nx)];
		}
	}
	if (threadIdx.x == blockDim.x-1){
			u_local[size_u_line*(pos_block_y+2) + (3+pos_block_x)] = u[((pos_y+2+nx)%nx)*nx + ((pos_x+3+nx)%nx)];
	}
	if(threadIdx.y == 0){
		if(threadIdx.x == blockDim.x-1){
			u_local[2*size_u_line-1] = u[((pos_y+1+nx)%nx)*nx + ((pos_x+3+nx)%nx)];
		}
		for(int j=0; j<2; j++){
				u_local[size_u_line*(pos_block_y+j) + (2+pos_block_x)] = u[((pos_y+j+nx)%nx)*nx + ((pos_x+2+nx)%nx)];
		}

	}
	if (threadIdx.y == blockDim.y-1){
			u_local[size_u_line*(pos_block_y+3) + (2+pos_block_x)] = u[((pos_y+3+nx)%nx)*nx + ((pos_x+2+nx)%nx)];
	}

	u_local[size_u_line*(pos_block_y+2) + (pos_block_x+2)] = u[(pos_y+2)*nx + (pos_x+2)];

	//data_3D
	if(threadIdx.x == 0){
			data_3D_local[(size_data_line*(pos_block_y+1) + (pos_block_x))*3] = data_3D_gpu[(((pos_data3D_y+1+nx)%nx)*nx + ((pos_data3D_x+nx)%nx))*3];
			data_3D_local[(size_data_line*(pos_block_y+1) + (pos_block_x))*3 + 1] = data_3D_gpu[(((pos_data3D_y+1+nx)%nx)*nx + ((pos_data3D_x+nx)%nx))*3 + 1];
			data_3D_local[(size_data_line*(pos_block_y+1) + (pos_block_x))*3 + 2] = data_3D_gpu[(((pos_data3D_y+1+nx)%nx)*nx + ((pos_data3D_x+nx)%nx))*3 + 2];
	}
	if(threadIdx.y == 0){
			data_3D_local[(size_data_line*(pos_block_y) + (pos_block_x + 1))*3] = data_3D_gpu[(((pos_data3D_y+nx)%nx)*nx + ((pos_data3D_x + 1 + nx)%nx))*3];
			data_3D_local[(size_data_line*(pos_block_y) + (pos_block_x + 1))*3 + 1] = data_3D_gpu[(((pos_data3D_y+nx)%nx)*nx + ((pos_data3D_x + 1 +nx)%nx))*3 + 1];
			data_3D_local[(size_data_line*(pos_block_y) + (pos_block_x + 1))*3 + 2] = data_3D_gpu[(((pos_data3D_y+nx)%nx)*nx + ((pos_data3D_x + 1 +nx)%nx))*3 + 2];
	}

	data_3D_local[(size_data_line*(pos_block_y+1) + (1+pos_block_x))*3] = data_3D_gpu[((pos_data3D_y+1)*nx + (pos_data3D_x+1))*3];
	data_3D_local[(size_data_line*(pos_block_y+1) + (1+pos_block_x))*3 + 1] = data_3D_gpu[((pos_data3D_y+1)*nx + (pos_data3D_x+1))*3 + 1];
	data_3D_local[(size_data_line*(pos_block_y+1) + (1+pos_block_x))*3 + 2] = data_3D_gpu[((pos_data3D_y+1)*nx + (pos_data3D_x+1))*3 + 2];

	//data_edge
	data_edge_local[(size_edge_line*(pos_block_y) + (pos_block_x))*2] = data_edge_gpu[((pos_y_data)*nx + (pos_x_data))*2];
	data_edge_local[(size_edge_line*(pos_block_y) + (pos_block_x))*2 + 1] = data_edge_gpu[((pos_y_data)*nx + (pos_x_data))*2 + 1];


	__syncthreads();

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

	for(int direction=0; direction<2; direction++){
		if(direction==0){ //horizontal
			i_p = -1;
			j_p = 0;
		} else { //vertical
			i_p = 0;
			j_p = -1;
		}
		if(direction == 0){
			lap_q = (u_local[size_u_line*(pos_block_j) + (pos_block_i+1)] + u_local[size_u_line*(pos_block_j+1) + (pos_block_i)] + u_local[size_u_line*(pos_block_j-1) + (pos_block_i)]);
			lap_p = (u_local[size_u_line*(pos_block_j+j_p) + (i_p+pos_block_i-1)] + u_local[size_u_line*(pos_block_j+j_p+1) + (i_p+pos_block_i)] + u_local[size_u_line*(pos_block_j+j_p-1) + (i_p+pos_block_i)]);
		} else {
			lap_q = (u_local[size_u_line*(pos_block_j) + (pos_block_i+1)] + u_local[size_u_line*(pos_block_j+1) + (pos_block_i)] + u_local[size_u_line*(pos_block_j) + (pos_block_i-1)]);
			lap_p = (u_local[size_u_line*(pos_block_j+j_p) + (i_p+pos_block_i-1)] + u_local[size_u_line*(pos_block_j+j_p) + (i_p+pos_block_i+1)] + u_local[size_u_line*(pos_block_j+j_p-1) + (i_p+pos_block_i)]);
		}

		u_p = u_local[size_u_line*(pos_block_j+j_p) + (i_p+pos_block_i)];
		u_q = u_local[size_u_line*(pos_block_j) + (pos_block_i)];

		H_p = data_3D_local[(size_data_line*(pos_data3D_j+j_p) + (i_p+pos_data3D_i))*3];
		H_q = data_3D_local[(size_data_line*(pos_data3D_j) + (pos_data3D_i))*3];

		T_p = data_3D_local[(size_data_line*(pos_data3D_j+j_p) + (i_p+pos_data3D_i))*3 + 1];
		T_q = data_3D_local[(size_data_line*(pos_data3D_j) + (pos_data3D_i))*3 + 1];

		ct_p = data_3D_local[(size_data_line*(pos_data3D_j+j_p) + (i_p+pos_data3D_i))*3 + 2];
		ct_q = data_3D_local[(size_data_line*(pos_data3D_j) + (pos_data3D_i))*3 + 2];

		k_E = data_edge_local[(size_edge_line*(pos_block_y) + (pos_block_x))*2];
		H_E = data_edge_local[(size_edge_line*(pos_block_y) + (pos_block_x))*2 + 1];

		W_q = G*(nx-(pos_y_data)-0.5f)*h - H_q;
		W_p = G*(nx-(pos_y_data+j_p)-0.5f)*h - H_p;




		M = 2.0f * u_q*u_q * u_p*u_p /(3.0f*(u_q + u_p)) + (e/6.0f)*u_q*u_q*u_p*u_p*(H_E+k_E);

		theta = h*h + (tau*M*(10.0f*e + 2.0f*eta + G*e*(ct_p + ct_q) - e*(T_p + T_q)));
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

		if(direction==0){
				flx_x[(pos_y_data)*nx + (pos_x_data)] = delta_u;
		} else {
				flx_y[(pos_y_data)*nx + (pos_x_data)] = delta_u;
		}
	}

}

__global__ void update_u(float *u, float* flux_x, float* flux_y, int nx){
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int i = k%nx;
	int j = k/nx;

	if(i == nx-1){
		if(j == nx-1){
			u[k] = u[k] + flux_x[k] + flux_y[k] - flux_x[nx*j] - flux_y[i];
		} else {
			u[k] = u[k] + flux_x[k] + flux_y[k] - flux_x[nx*j] - flux_y[nx*(j+1)+i];
		}
	} else if(j == nx-1){
		u[k] = u[k] + flux_x[k] + flux_y[k] - flux_x[nx*j+i+1] - flux_y[i];
	} else {
		u[k] = u[k] + flux_x[k] + flux_y[k] - flux_x[nx*j+i+1] - flux_y[nx*(j+1)+i];
	}
}
