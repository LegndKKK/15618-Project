#include "CycleTimer.h"
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <unordered_map>
#include <chrono>

#define PARTICLE_NUM 8192
#define NEIGHBOR_NUM 300
#define OMP 1
#define CUDA 1

#define Debug 1
#if DEBUG
#define dbg_printf(...) printf(__VA_ARGS__)
#else
#define dbg_printf(...)
#endif

// --------------------------------------------------------------------
struct globalConstants
{
    const float G = .02f * .25;        // Gravitational Constant for our simulation
    const float spacing = 2.f;         // Spacing of particles
    const float k = spacing / 1000.0f; // Far pressure weight
    const float k_near = k * 10;       // Near pressure weight
    const float rest_density = 3.f;      // Rest Density
    const float r = spacing * 1.25f;   // Radius of Support
    const float rsq = r * r;           // ... squared for performance stuff
    const float SIM_W = 50;            // The size of the world
    const float bottom = 0;            // The floor of the world
    const float sigma = 3.f;
    const float beta = 4.f;
};
const globalConstants constantParams;

// A structure for holding two neighboring particles and their weighted distances
struct Particle;
struct Neighbor
{
    Particle *j;
    float q, q2;
};

// The Particle structure holding all of the relevant information.
struct Particle
{
    glm::vec2 pos;
    float r, g, b;
    glm::vec2 pos_old;
    glm::vec2 vel;
    glm::vec2 force;
    float mass;
    float rho;
    float rho_near;
    float press;
    float press_near;
    std::vector<Neighbor> neighbors;

    // for CUDA use
    int id;
};

void Surprise_CUDA(long N, Particle *particles, int *neighborNum, int *neighborIndex, float *neighborDist);
