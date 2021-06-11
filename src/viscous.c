#include "viscous.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void initialization(float* u, int nx, int ny,  float h, int choice){

  for(int i=0; i<nx*ny; i++){
    u[i] = 0.04f; // 0.0001
	}

  // GAUSSIANS
  if(choice == 1){

    gaussians(u, nx, ny, h);

  }
  // Center circle + line
  else if(choice == 2) {
    circle(u, nx, ny, 0.04f);
  }

  else if(choice == 3) {
    simple_gaussian(u, nx, ny, h);
  }

}

void gaussians(float* u, int nx, int ny, float h){

  float mu_x[5] = {0.19f, 0.2f, 0.56f, 0.6f, 0.9f};
	float mu_y[5] = {0.8f, 0.45f, 0.7f, 0.3f, 0.5f};
	float sigma_x[5] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
	float sigma_y[5] = {0.07f, 0.07f, 0.07f, 0.07f, 0.07f};
	float  x, y;
  float density;
	int i, j;
	float max = 0.0f;


	for(int l=0; l<5; l++){
		for(int index=0 ; index<nx*ny ; index++){
			i = (int) index % nx;
			j = (int) index / nx;

			x = i*h;
			y = j*h;

			density = (float)(1.0f/(100.0f*2.0f*M_PI*sigma_x[l]*sigma_y[l])) * exp(-(1.0f/2.0f)*((x-mu_x[l])*(x-mu_x[l])/(sigma_x[l]*sigma_x[l]) + (y-mu_y[l])*(y-mu_y[l])/(sigma_y[l]*sigma_y[l])));
			if (density > u[index]){
				u[index] = density;
				if (density > max){
					max = density;
				}
			}
		}
	}
  printf("Max = %f \n", max);

}

void circle(float* u, int nx, int ny, float value){
  for(int i=0; i<nx*ny; i++){
    if((i/nx - 256)*(i/nx - 256) + (i%nx -256)*(i%nx - 256) < 75*75){
    	u[i] = value;
    }
  }
}

void float_circle(float* u, int nx, int ny, float value){
  for(int i=0; i<nx*ny; i++){
    if((((i/nx - 150)*(i/nx - 150) + (i%nx -170)*(i%nx - 170) < 75*75) && i/nx > 175) || (((i/nx - 150)*(i/nx - 150) + (i%nx -342)*(i%nx - 342) < 75*75) && i/nx > 175)){
			u[i] = value;
		}
  }
}

void simple_gaussian(float* u, int nx, int ny, float h){
  float mu_x[1] = {0.5f};
	float mu_y[1] = {0.15f};
	float sigma_x[1] = {0.1f};
	float sigma_y[1] = {0.07f};
	float density, x, y;
	int i, j;
	float max = 0;

	for(int l=0; l<sizeof(mu_x); l++){
		for(int index=0 ; index<nx*ny ; index++){
			i = (int) index % nx;
			j = (int) index / nx;

			x = i*h;
			y = j*h;

			density = (1.0f/(50.0f*2.0f*M_PI*sigma_x[l]*sigma_y[l])) * exp(-(1.0f/2.0f)*((x-mu_x[l])*(x-mu_x[l])/(sigma_x[l]*sigma_x[l]) + (y-mu_y[l])*(y-mu_y[l])/(sigma_y[l]*sigma_y[l])));
			if (density > u[index]){
				u[index] = density;
				if (density > max){
					max = density;
				}
			}
		}
	}
	printf("Max = %f \n", max);
}

void big_line(float* u, int nx, int ny, float value){
  for(int index=350*nx ; index<390*nx ; index++){
		u[index] = value;
	}
}

void merging_gaussian(float* u, int nx, int ny, float h){
  float mu_x[2] = {0.5f, 0.65f};
	float mu_y[2] = {0.6f, 0.8f};
	float sigma_x[2] = {0.1f, 0.1f};
	float sigma_y[2] = {0.1f, 0.1f};
	float density, x, y;
	int i, j;
	float max = 0.0f;

	for(int l=0; l<sizeof(mu_x); l++){
		for(int index=0 ; index<nx*ny ; index++){
			i = (int) index % nx;
			j = (int) index / nx;

			x = i*h;
			y = j*h;

			density = (1.0f/(100.0f*2.0f*M_PI*sigma_x[l]*sigma_y[l])) * exp(-(1.0f/2.0f)*((x-mu_x[l])*(x-mu_x[l])/(sigma_x[l]*sigma_x[l]) + (y-mu_y[l])*(y-mu_y[l])/(sigma_y[l]*sigma_y[l])));
			if (density > u[index]){
				u[index] = density;
				if (density > max){
					max = density;
				}
			}
		}
	}
	printf("Max = %f \n", max);

}


float x_deriv(float h_r, float h_l, float dx){
  return (h_r-h_l)/(2.0f*dx);
}

float y_deriv(float h_u, float h_d, float dy){
  return (h_u-h_d)/(2.0f*dy);
}

float xx_deriv(float h_r, float h_l, float h, float dx){
  return (h_r - 2.0f*h + h_l)/(dx*dx);
}

float yy_deriv(float h_u, float h_d, float h, float dy){
  return (h_u - 2.0f*h + h_d)/(dy*dy);
}

float xy_deriv(float h_ur, float h_dr, float h_ul, float h_dl, float dx, float dy){
  return (h_ur - h_dr - h_ul + h_dl)/(4.0f*dx*dy);
}

void init_surface_height_map(float* H, float* T, float* ctheta, float* height, int nx, int ny, float dx){
  float h_ul, h_u, h_ur, h_l, h, h_r, h_dl, h_d, h_dr;
  float h_x, h_xx, h_y, h_yy, h_xy;
  float K;
  for (int j=1; j<ny-1; j++){
    for(int i=1; i<nx-1; i++){
      h = height[nx*j+i];
      h_ul = height[nx*(j+1) + i-1];
      h_l = height[nx*(j) + i-1];
      h_dl = height[nx*(j-1) + i-1];
      h_u = height[nx*(j+1) + i];
      h_d = height[nx*(j-1) + i];
      h_ur = height[nx*(j+1) + i+1];
      h_r = height[nx*(j) + i+1];
      h_dr = height[nx*(j-1) + i+1];

      h_x = x_deriv(h_r, h_l, dx);
      h_y = y_deriv(h_u, h_d, dx);
      h_xx = xx_deriv(h_r, h_l, h, dx);
      h_yy = yy_deriv(h_u, h_d, h, dx);
      h_xy = xy_deriv(h_ur, h_dr, h_ul, h_dl, dx, dx);

      H[nx*j + i] = (h_xx*(1.0f+h_y*h_y) + h_yy*(1.0f+h_x*h_x) - 2.0f*h_xy*h_x*h_y) / (2.0f * pow((1.0f+h_x*h_x+h_y*h_y), 1.5f));
      K = (h_xx*h_yy - h_xy*h_xy) / (pow(1.0f+h_x*h_x+h_y*h_y, 2.0f));
      T[nx*j + i] = H[nx*j + i]*H[nx*j + i] - 2.0f*K ;
      ctheta[nx*j + i] = -h_y/(pow(1+h_x*h_x+h_y*h_y, 0.5f));
    }
  }

  for(int i=0; i<nx; i++){ //cas j=0
    H[i] = H[nx+i];
    T[i] = T[nx+i];
    ctheta[i] = ctheta[nx+i];
  }

  for(int i=0; i<nx; i++){ //cas j=ny-1
    H[nx*(ny-1) + i] = H[i];
    T[nx*(ny-1) + i] = T[i];
    ctheta[nx*(ny-1) + i] = ctheta[i];
  }

  for(int j=0; j<ny; j++){ //cas i=0
    H[nx*j] = H[nx*j+1];
    T[nx*j] = T[nx*j+1];
    ctheta[nx*j] = ctheta[nx*j+1];
  }

  for(int j=0; j<ny; j++){ //cas i=nx-1
    H[nx*j+ nx-1] = H[nx*j];
    T[nx*j + nx-1] = T[nx*j];
    ctheta[nx*j + nx-1] = ctheta[nx*j];
  }

}

void init_height_map_edge(float* H_edge_x, float* H_edge_y, float* k_x, float* k_y, float* height_x_edge, float* height_y_edge, int nx, int ny, float dx){
  float h_ul, h_u, h_ur, h_l, h, h_r, h_dl, h_d, h_dr;
  float h_x, h_xx, h_y, h_yy, h_xy;

  // x edge
  for (int j=1; j<ny-1; j++){
    for(int i=1; i<nx; i++){
      h = height_x_edge[(nx+1)*j+i];
      h_ul = height_x_edge[(nx+1)*(j+1) + i-1];
      h_l = height_x_edge[(nx+1)*(j) + i-1];
      h_dl = height_x_edge[(nx+1)*(j-1) + i-1];
      h_u = height_x_edge[(nx+1)*(j+1) + i];
      h_d = height_x_edge[(nx+1)*(j-1) + i];
      h_ur = height_x_edge[(nx+1)*(j+1) + i+1];
      h_r = height_x_edge[(nx+1)*(j) + i+1];
      h_dr = height_x_edge[(nx+1)*(j-1) + i+1];

      h_x = x_deriv(h_r, h_l, dx);
      h_y = y_deriv(h_u, h_d, dx);
      h_xx = xx_deriv(h_r, h_l, h, dx);
      h_yy = yy_deriv(h_u, h_d, h, dx);
      h_xy = xy_deriv(h_ur, h_dr, h_ul, h_dl, dx, dx);

      H_edge_x[(nx+1)*j + i] = (h_xx*(1.0f+h_y*h_y) + h_yy*(1.0f+h_x*h_x) - 2.0f*h_xy*h_x*h_y) / (2.0f * pow((1.0f+h_x*h_x+h_y*h_y), 1.5f));
      k_x[(nx+1)*j + i] = (h_xx)/(pow((1.0f + h_x*h_x + h_y*h_y),0.5f));
    }
  }
  //y edge
  for (int j=1; j<ny; j++){
    for(int i=1; i<nx-1; i++){
      h = height_y_edge[(nx)*j+i];
      h_ul = height_y_edge[(nx)*(j+1) + i-1];
      h_l = height_y_edge[(nx)*(j) + i-1];
      h_dl = height_y_edge[(nx)*(j-1) + i-1];
      h_u = height_y_edge[(nx)*(j+1) + i];
      h_d = height_y_edge[(nx)*(j-1) + i];
      h_ur = height_y_edge[(nx)*(j+1) + i+1];
      h_r = height_y_edge[(nx)*(j) + i+1];
      h_dr = height_y_edge[(nx)*(j-1) + i+1];

      h_x = x_deriv(h_r, h_l, dx);
      h_y = y_deriv(h_u, h_d, dx);
      h_xx = xx_deriv(h_r, h_l, h, dx);
      h_yy = yy_deriv(h_u, h_d, h, dx);
      h_xy = xy_deriv(h_ur, h_dr, h_ul, h_dl, dx, dx);

      H_edge_y[(nx)*j + i] = (h_xx*(1.0f+h_y*h_y) + h_yy*(1.0f+h_x*h_x) - 2.0f*h_xy*h_x*h_y) / (2.0f * pow((1.0f+h_x*h_x+h_y*h_y), 1.5f));
      k_y[(nx)*j + i] = (h_yy)/(pow((1.0f + h_x*h_x + h_y*h_y),0.5f));
    }
  }
  for(int i=0; i<nx; i++){ //cas j=0 pour y
    H_edge_y[i] = H_edge_y[(nx)+i];
    k_y[i] = k_y[(nx)+i];
  }
  for(int i=0; i<nx+1; i++){ //cas j=0 pour x
    H_edge_x[i] = H_edge_x[(nx+1)+i];
    k_x[i] = k_x[(nx+1)+i];
  }

  for(int i=0; i<nx; i++){ //cas j=ny pour y
    H_edge_y[nx*(ny) + i] = H_edge_y[i];
    k_y[nx*(ny) + i] = k_y[i];
  }
  for(int i=0; i<nx+1; i++){ //cas j=ny-1 pour x
    H_edge_x[(nx+1)*(ny-1) + i] = H_edge_x[i];
    k_x[(nx+1)*(ny-1) + i] = k_x[i];
  }

  for(int j=0; j<ny+1; j++){ //cas i=0 pour y
    H_edge_y[nx*j] = H_edge_y[nx*j+1];
    k_y[nx*j] = k_y[nx*j+1];
  }
  for(int j=0; j<ny; j++){ //cas i=0 pour x
    H_edge_x[(nx+1)*j] = H_edge_x[(nx+1)*j+1];
    k_x[(nx+1)*j] = k_x[(nx+1)*j+1];
  }

  for(int j=0; j<ny+1; j++){ //cas i=nx-1 pour y
    H_edge_y[nx*j+ nx-1] = H_edge_y[nx*j];
    k_y[nx*j+ nx-1] = k_y[nx*j];
  }
  for(int j=0; j<ny; j++){ //cas i=nx pour x
    H_edge_x[(nx+1)*j+ nx] = H_edge_x[(nx+1)*j];
    k_x[(nx+1)*j+ nx] = k_x[(nx+1)*j];
  }
}

void read_txt(float* height, float* height_x_edge, float* height_y_edge, char *fileName, int nx){
  int j;
  int counter;
  float test;
  int dummy;
  float deepness = 0.000002f;
  FILE *file;
  file = fopen(fileName,"r");
  if (file == NULL){
      exit(1);
  }
  j=0;
  counter = 0;
  while (feof(file) == 0)
  {
    for(int i=0; i<nx; i++){
      if(counter%2 == 0){
        dummy = fscanf(file, "%f, ", &test);
        dummy = fscanf(file, "%f, ", &height_y_edge[nx*j+i]);
        height_y_edge[nx*j+i] = deepness*height_y_edge[nx*j+i];
      } else{
        dummy = fscanf(file, "%f, ", &height_x_edge[(nx+1)*j+i]);
        height_x_edge[(nx+1)*j+i] = deepness * height_x_edge[(nx+1)*j+i];
        dummy = fscanf(file, "%f, ", &height[nx*j+i]);
        height[nx*j+i] = deepness * height[nx*j+i];
      }
    }
    if(counter%2==0){
      dummy = fscanf(file, "%f, ", &test);
    } else {
      dummy = fscanf(file, "%f, ", &height_x_edge[(nx+1)*j+nx]);
      height_x_edge[(nx+1)*j+nx] = deepness * height_x_edge[(nx+1)*j+nx];
    }

    if(counter%2 == 1){
      j++;
    }
    counter++;
    //printf("counter = %d \n", counter);

  }
  fclose(file);
}
