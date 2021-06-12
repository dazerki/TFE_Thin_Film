Simulation de films fins en temps réel.

Ce projet simule l'équation du film fin en temps réel suivante:

<a href="https://www.codecogs.com/eqnedit.php?latex=\partial_t&space;u&space;&plus;&space;\nabla&space;\cdot&space;\hat{f}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\partial_t&space;u&space;&plus;&space;\nabla&space;\cdot&space;\hat{f}&space;=&space;0" title="\partial_t u + \nabla \cdot \hat{f} = 0" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{f}&space;=&space;M(u)&space;\nabla&space;\left(-\zeta&space;z&space;-&space;\epsilon&space;\Delta&space;u&space;&plus;&space;\eta&space;u&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{f}&space;=&space;M(u)&space;\nabla&space;\left(-\zeta&space;z&space;-&space;\epsilon&space;\Delta&space;u&space;&plus;&space;\eta&space;u&space;\right)" title="\hat{f} = M(u) \nabla \left(-\zeta z - \epsilon \Delta u + \eta u \right)" /></a>



## Implémentation C et CUDA

Les branches CPU_2D et CPU_3D exécutent un code C en utilisant OpenMP tandis que les autres branches utilisent CUDA.

La procédure de compilation et d'exécution consiste en:

```bash
cd /path/to/project
mkdir build
cd build
cmake ..
make
./project
```
Normalement, une fenêtre devrait s'ouvrir et résoudre l'équation du film en temps réel. Il est alors possible d'interagir avec la simulation en réalisant un clic gauche. L'utilisateur peut également choisir les paramètres de la simulation grâce à des "flags" à rajouter à la commande "cmake":
  * `-DN_DISCR = 512` : Discrétisation de la grille NxN, ce paramètre est uniquement disponible pour les simulations 2D et  Gauss_Seidel_Simplifie
  * `-DZETA = 5.0f` : Nombre de Bond de la simulation, ce nombre devrait se situer entre 13 et 2 en général. Pour assurer l'efficacité de la simulation, il faut indiquer le nombre sous format float en simple précision: 5.0f
  * `-DEPSILON = 0.01f` : Facteur géométrique hauteur du film/ longueur d'onde.
  * `-DETA = 0.0f` : Facteur stabilisation
  * `-DDELTA_T = 0.0001f` : Pas de temps de la discrétisation spatiale.

## Instructions d'installation

Les instructions suivantes correspondent au manager de packet `apt` sur Linux, cependant cela ne devrait pas trop différer d'autres méthodes.

### Visualisation (OpenGL & GLFW3)

La visualisation est réalisée grâce à OpenGL & GLFW3. Il est donc nécessaire d'avoir OpenGL, GLFW3 et GLEW installés sur votre système:
```bash
sudo apt-get install freeglut3-dev libglfw3-dev  libglew-dev
```

### CUDA

Pour exécuter les codes des branches `GPU_*`, vous aurez besoin d'un processeur graphique de Nvidia et d'installer CUDA. Cette installation délicate varie d'un ordinateur à l'autre, la procédure à suivre est indiquée dans le [guide officiel d'installation.](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

### OpenMP

Pour exécuter les codes des branches `CPU_*`, vous aurez besoin de OpenMP. Son installation consiste en:

```bash
sudo apt-get install libomp-dev
```
