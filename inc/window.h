#ifndef _WINDOW_H_
#define _WINDOW_H_

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>


GLFWwindow *init_window();

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

void add_fluid(GLFWwindow* window, float* u);

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

GLboolean getdrag();


#endif
