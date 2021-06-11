#include "viscous.h"
#include "window.h"
#include "shaders.h"

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>



float* u;
GLboolean drag=0;


int parity(int di, int dj, int i, int j, int rho);
void add_fluid(GLFWwindow* window);

int main(int argc, char *argv[]){

	int nx = 512;
	int ny = 512;
	float h = 1.0f/nx ;

	// memory allocation
	u = (float*)calloc(nx*ny, sizeof(float));

	//init
	initialization(u, nx, ny, h, 3);

	//Initialise window
  GLFWwindow *window = init_window();

  // Initialise shaders
  init_shaders();

  // Create Vertex Array Object
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // Create a Vertex Buffer Object for positions
  GLuint vbo_pos;
  glGenBuffers(1, &vbo_pos);

	GLfloat *positions = (GLfloat*) malloc(2*nx*nx*sizeof(GLfloat));
  for (int i = 0; i < nx; i++) {
      for (int j = 0; j < nx; j++) {
          int ind = j*nx+i;
          positions[2*ind  ] = (float)(1.0 - 2.0*i/(nx-1));
          positions[2*ind+1] = (float)(1.0 - 2.0*j/(nx-1));
      }
  }

  glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
  glBufferData(GL_ARRAY_BUFFER, 2*nx*nx*sizeof(GLfloat), positions, GL_STATIC_DRAW);

  // Specify vbo_pos' layout
  GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
  glEnableVertexAttribArray(posAttrib);
  glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

  // Create an Element Buffer Object and copy the element data to it
  GLuint ebo;
  glGenBuffers(1, &ebo);

	GLuint *elements = (GLuint*) malloc(4*(nx-1)*(nx-1)*sizeof(GLuint));
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < nx-1; j++) {
            int ind  = i*nx+j;
            int ind_ = i*(nx-1)+j;

            elements[4*ind_  ] = ind;
            elements[4*ind_+1] = ind+1;
            elements[4*ind_+2] = ind+nx;
            elements[4*ind_+3] = ind+nx+1;
        }
    }

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*(nx-1)*(nx-1)*sizeof(GLuint), elements, GL_STATIC_DRAW);

	// Create a Vertex Buffer Object for colors
  GLuint vbo_colors;
  glGenBuffers(1, &vbo_colors);

  GLfloat *colors = (GLfloat*) malloc(nx*nx*sizeof(GLfloat));
  for (int i = 0; i < nx; i++) {
      for (int j = 0; j < nx; j++) {
          int ind = i*nx+j;
          colors[ind] = (float) u[ind];
      }
  }

  glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  glBufferData(GL_ARRAY_BUFFER, nx*nx*sizeof(GLfloat), colors, GL_STREAM_DRAW);

  // Specify vbo_color's layout
  GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
  glEnableVertexAttribArray(colAttrib);
  glVertexAttribPointer(colAttrib, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);



	// PARAMETER
	float tau = DELTA_T ;
	float e = EPSILON;
	float eta = ETA;
	float G = ZETA;
	int n_passe = 30;


 	omp_set_num_threads(6);

	while(!glfwWindowShouldClose(window)) {

		for(int p=0; p<n_passe; p++){

			//Flux in direcion (di, dj) = (1,0) Horizontal
			int di = 1;
			int dj = 0;

			for(int rho=0; rho<4; rho++){
				#pragma omp parallel for
				for(int k=0; k<nx*ny; k++){
					int rho_ij, i_p, j_p;
					float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
					int i,j;
					float val;
					float u_p, u_q;


					i = (int) k % nx;
					j = (int) k / nx;

					rho_ij = parity(di, dj, i, j, rho);

					if (rho_ij == 3){
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

						//3D
						theta = h*h + (tau*M*(10.0f*e + 2.0f*eta));
						f = (M*h/(theta)) * (eta*(u_p - u_q) + e*(lap_q - lap_p + 5.0f*(u_p-u_q)) + W_p-W_q);

						val = tau*f/h;
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
			}

			//Flux in direcion (di, dj) = (0,1) Vertical
			di = 0;
			dj = 1;

			for(int rho=0; rho<4; rho++){
				#pragma omp parallel for
				for(int k=0; k<nx*ny; k++){
					int rho_ij, i_p, j_p;
					float W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
					int i,j;
					float val;
					float u_p, u_q;

					i = (int) k % nx;
					j = (int) k / nx;

					rho_ij = parity(di, dj, i, j, rho);
					if (rho_ij == 3){
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

						//nouveau
						W_q = G*(ny-j-0.5f)*h;

						if(j==0){
							W_p = G*(ny-(-1.0f)-0.5f)*h;
						}else{
							W_p = G*(ny-j_p-0.5f)*h;
						}

						M = 2.0f * u_q*u_q * u_p*u_p /(3.0f*(u_q + u_p));

						theta = h*h + (tau*M*(10.0f*e + 2.0f*eta));
						f = (M*h/(theta)) * (eta*(u_p - u_q) + e*(lap_q - lap_p + 5.0f*(u_p-u_q)) + W_p-W_q);

						val = tau*f/h;
						if(u_p<val){
							delta_u = u_p;
						} else{
							if(val > -u_q){
								delta_u = val;
							} else {
								delta_u = -u_q;
							}
						}

						u[nx*j + i] = u[nx*j + i] + delta_u;
						u[nx*j_p + i_p] = u[nx*j_p + i_p] - delta_u;
				}
			}
		}
		glfwPollEvents();
		if(drag){
			add_fluid(window);
		}
	}

	glfwSwapBuffers(window);


	// Clear the screen to black
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < nx*nx; i++) {
			colors[i] = (float) (u[i]);
	}

	glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
	glBufferData(GL_ARRAY_BUFFER, nx*nx*sizeof(GLfloat), colors, GL_STREAM_DRAW);


	// Draw elements
	glDrawElements(GL_LINES_ADJACENCY, 4*(nx-1)*(nx-1), GL_UNSIGNED_INT, 0);

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, GL_TRUE);



}

	//free memory
	free(u);

	printf("\n *Happy computer sound* \n");

	return 0;
}


int parity(int di, int dj, int i, int j, int rho){
	return ((dj+1)*i + (di+1)*j + rho) % 4;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{

		if(button == GLFW_MOUSE_BUTTON_LEFT) {
			drag = (action == GLFW_PRESS);
		}

}

GLFWwindow *init_window() {
    // Init GLFW & window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    GLFWwindow* window = glfwCreateWindow(800, 800, "Viscous film", NULL, NULL);
    glfwMakeContextCurrent(window);

    // Callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Init GLEW
    glewExperimental = GL_TRUE;
    glewInit();

    return window;
}

void add_fluid(GLFWwindow* window){
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	int i = 512-floor(512*xpos/800);
	int j = floor(512*ypos/800);
	for(int k=-20; k<20; k++){
		for(int p=-20; p<20 ; p++){
			if((k*k)+(p*p)<400){
				u[512*(j+p)+(i+k)] = u[512*(j+p)+(i+k)] + 0.002f;
			}
		}
	}
}
