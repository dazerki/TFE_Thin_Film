#include "window.h"
#include <math.h>


/*
 *  GLFW, GLEW initialisation
 */

 GLboolean drag = 0;
 /*
  *  GLFW, GLEW initialisation
  */
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


 void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {

 		if(button == GLFW_MOUSE_BUTTON_LEFT) {
 			drag = (action == GLFW_PRESS);
 		}

 }

 void add_fluid(GLFWwindow* window, float* u){
 	double xpos, ypos;
 	glfwGetCursorPos(window, &xpos, &ypos);
 	int i = N_DISCR-floor(N_DISCR*xpos/800);
 	int j = floor(N_DISCR*ypos/800);
 	for(int k=-20; k<20; k++){
 		for(int p=-20; p<20 ; p++){
 			if((k*k)+(p*p)<N_DISCR-100){
 				u[N_DISCR*(j+p)+(i+k)] = u[N_DISCR*(j+p)+(i+k)] + 0.002f;
 			}
 		}
 	}
 }

 GLboolean getdrag(){
   return drag;
 }
