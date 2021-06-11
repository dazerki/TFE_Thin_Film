#ifndef _VISCOUS_H_
#define _VISCOUS_H_

void initialization(float* u, int nx, int ny, float h, int choice);

void float_circle(float* u, int nx, int ny, float value);

void circle(float* u, int nx, int ny, float value);

void gaussians(float* u, int nx, int ny, float h);

void simple_gaussian(float* u, int nx, int ny, float h);

void big_line(float* u, int nx, int ny, float value);

void merging_gaussian(float* u, int nx, int ny, float h);

void init_surface_height_map(float* H, float* T, float* ctheta, float* height, int nx, int ny, float dx);

float x_deriv(float h_r, float h_l, float dx);

float y_deriv(float h_u, float h_d, float dy);

float xx_deriv(float h_r, float h_l, float h, float dx);

float yy_deriv(float h_u, float h_d, float h, float dy);

float xy_deriv(float h_ur, float h_dr, float h_ul, float h_dl, float dx, float dy);

void read_txt(float* height,float* height_x_edge, float* height_y_edge, char *fileName, int nx);

void init_height_map_edge(float* H_edge_x, float* H_edge_y, float* k_x, float* k_y, float* height_x_edge, float* height_y_edge, int nx, int ny, float dx);

#endif
