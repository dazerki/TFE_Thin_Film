#include "window.h"
#include <math.h>


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
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Init GLEW
    glewExperimental = GL_TRUE;
    glewInit();

    return window;
}



/*
 *  Callback for key presses
 */
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        printf("Spacebar pressed !\n");
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
		if(button == GLFW_MOUSE_BUTTON_LEFT) {
			drag = (action == GLFW_PRESS);
		}

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
