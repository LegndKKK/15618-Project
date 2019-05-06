// Real-Time Physics Tutorials
// Brandon Pelfrey
// SPH Fluid Simulation
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <typeinfo>


#define parallel 1

#define idxv 1

#define DEBUG 1

#if DEBUG
#define dbg_print(...) fprintf(stdout, __VA_ARGS__)
#endif

#ifndef _SYRAH_CYCLE_TIMER_H_
#define _SYRAH_CYCLE_TIMER_H_

#if defined(__APPLE__)
  #if defined(__x86_64__)
    #include <sys/sysctl.h>
  #else
    #include <mach/mach.h>
    #include <mach/mach_time.h>
  #endif // __x86_64__ or not

  #include <stdio.h>  // fprintf
  #include <stdlib.h> // exit

#elif _WIN32
#  include <windows.h>
#  include <time.h>
#else
#  include <stdio.h>
#  include <stdlib.h>
#  include <string.h>
#  include <sys/time.h>
#endif



class CycleTimer {
public:
typedef unsigned long long SysClock;

    //////////
    // Return the current CPU time, in terms of clock ticks.
    // Time zero is at some arbitrary point in the past.
    static SysClock currentTicks() {
#if defined(__APPLE__) && !defined(__x86_64__)
      return mach_absolute_time();
#elif defined(_WIN32)
      LARGE_INTEGER qwTime;
      QueryPerformanceCounter(&qwTime);
      return qwTime.QuadPart;
#elif defined(__x86_64__)
      unsigned int a, d;
      asm volatile("rdtsc" : "=a" (a), "=d" (d));
      return static_cast<unsigned long long>(a) |
        (static_cast<unsigned long long>(d) << 32);
#elif defined(__ARM_NEON__) && 0 // mrc requires superuser.
      unsigned int val;
      asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(val));
      return val;
#else
      timespec spec;
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &spec);
      return CycleTimer::SysClock(static_cast<float>(spec.tv_sec) * 1e9 + static_cast<float>(spec.tv_nsec));
#endif
    }

    //////////
    // Return the current CPU time, in terms of seconds.
    // This is slower than currentTicks().  Time zero is at
    // some arbitrary point in the past.
    static double currentSeconds() {
      return currentTicks() * secondsPerTick();
    }

    //////////
    // Return the conversion from seconds to ticks.
    static double ticksPerSecond() {
      return 1.0/secondsPerTick();
    }

    static const char* tickUnits() {
#if defined(__APPLE__) && !defined(__x86_64__)
      return "ns";
#elif defined(__WIN32__) || defined(__x86_64__)
      return "cycles";
#else
      return "ns"; // clock_gettime
#endif
    }

    //////////
    // Return the conversion from ticks to seconds.
    static double secondsPerTick() {
      static bool initialized = false;
      static double secondsPerTick_val;
      if (initialized) return secondsPerTick_val;
#if defined(__APPLE__)
  #ifdef __x86_64__
      int args[] = {CTL_HW, HW_CPU_FREQ};
      unsigned int Hz;
      size_t len = sizeof(Hz);
      if (sysctl(args, 2, &Hz, &len, NULL, 0) != 0) {
         fprintf(stderr, "Failed to initialize secondsPerTick_val!\n");
         exit(-1);
      }
      secondsPerTick_val = 1.0 / (double) Hz;
  #else
      mach_timebase_info_data_t time_info;
      mach_timebase_info(&time_info);

      // Scales to nanoseconds without 1e-9f
      secondsPerTick_val = (1e-9*static_cast<double>(time_info.numer))/
        static_cast<double>(time_info.denom);
  #endif // x86_64 or not
#elif defined(_WIN32)
      LARGE_INTEGER qwTicksPerSec;
      QueryPerformanceFrequency(&qwTicksPerSec);
      secondsPerTick_val = 1.0/static_cast<double>(qwTicksPerSec.QuadPart);
#else
      FILE *fp = fopen("/proc/cpuinfo","r");
      char input[1024];
      if (!fp) {
         fprintf(stderr, "CycleTimer::resetScale failed: couldn't find /proc/cpuinfo.");
         exit(-1);
      }
      // In case we don't find it, e.g. on the N900
      secondsPerTick_val = 1e-9;
      while (!feof(fp) && fgets(input, 1024, fp)) {
        // NOTE(boulos): Because reading cpuinfo depends on dynamic
        // frequency scaling it's better to read the @ sign first
        float GHz, MHz;
        if (strstr(input, "model name")) {
          char* at_sign = strstr(input, "@");
          if (at_sign) {
            char* after_at = at_sign + 1;
            char* GHz_str = strstr(after_at, "GHz");
            char* MHz_str = strstr(after_at, "MHz");
            if (GHz_str) {
              *GHz_str = '\0';
              if (1 == sscanf(after_at, "%f", &GHz)) {
                //printf("GHz = %f\n", GHz);
                secondsPerTick_val = 1e-9f / GHz;
                break;
              }
            } else if (MHz_str) {
              *MHz_str = '\0';
              if (1 == sscanf(after_at, "%f", &MHz)) {
                //printf("MHz = %f\n", MHz);
                secondsPerTick_val = 1e-6f / GHz;
                break;
              }
            }
          }
        } else if (1 == sscanf(input, "cpu MHz : %f", &MHz)) {
          //printf("MHz = %f\n", MHz);
          secondsPerTick_val = 1e-6f / MHz;
          break;
        }
      }
      fclose(fp);
#endif

      initialized = true;
      return secondsPerTick_val;
    }

    //////////
    // Return the conversion from ticks to milliseconds.
    static double msPerTick() {
      return secondsPerTick() * 1000.0;
    }

  private:
    CycleTimer();
  };

#endif // #ifndef _SYRAH_CYCLE_TIMER_H_


// --------------------------------------------------------------------
// Between [0,1]
float rand01()
{
    return (float)rand() * (1.f / RAND_MAX);
}

// --------------------------------------------------------------------
// Between [a,b]
float randab(float a, float b)
{
    return a + (b - a) * rand01();
}

// --------------------------------------------------------------------
// A structure for holding two neighboring particles and their weighted distances
struct Particle;
struct Neighbor
{
#if idxv
    int j;
#else
    Particle *j;
#endif
    
    float q;
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
    float sigma;
    float beta;
    std::vector<Neighbor> neighbors;
};


// Our collection of particles
std::vector<Particle> particles;

// --------------------------------------------------------------------
const float G = .02f * .25;        // Gravitational Constant for our simulation
const float spacing = .2f;         // Spacing of particles
const float k = spacing / 1000.0f; // Far pressure weight
const float k_near = k * 10;       // Near pressure weight
const float rest_density = 3;      // Rest Density
const float r = spacing * 1.25f;   // Radius of Support
const float rsq = r * r;           // ... squared for performance stuff
const float SIM_W = 50;            // The size of the world
const float bottom = 0;            // The floor of the world

// --------------------------------------------------------------------
void init(const unsigned int N)
{
    // Initialize particles
    // We will make a block of particles with a total width of 1/4 of the screen.
    float w = SIM_W / 4;
    for (float y = bottom + 1; y <= 100000; y += r * 0.5f)
    {
        for (float x = -w; x <= w; x += r * 0.5f)
        {
            if (particles.size() > N)
            {
                break;
            }

            Particle p;
            p.pos = glm::vec2(x, y);
            p.pos_old = p.pos + 0.001f * glm::vec2(rand01(), rand01());
            p.force = glm::vec2(0, 0);
            p.sigma = 3.f;
            p.beta = 4.f;
            particles.push_back(p);
        }
    }
}

// Mouse attractor
glm::vec2 attractor(999, 999);
bool attracting = false;

// --------------------------------------------------------------------
template <typename T>
class SpatialIndex
{
  public:
#if idxv
    typedef std::vector<int> NeighborList;
#else
    typedef std::vector<T *> NeighborList;
#endif

    SpatialIndex(
        const unsigned int numBuckets, // number of hash buckets
        const float cellSize,          // grid cell size
        const bool twoDeeNeighborhood  // true == 3x3 neighborhood, false == 3x3x3
        )
        : mHashMap(numBuckets), mInvCellSize(1.0f / cellSize)
    {
        // initialize neighbor offsets
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                if (twoDeeNeighborhood)
                    mOffsets.push_back(glm::ivec3(i, j, 0));
                else
                    for (int k = -1; k <= 1; k++)
                        mOffsets.push_back(glm::ivec3(i, j, k));
    }

#if idxv
    void Insert_idx(const glm::vec3 &pos, int idx)
    {
        //std::cout<<typeid(thing).name()<<std::endl;
        mHashMap[Discretize(pos, mInvCellSize)].push_back(idx);
    }
#else

    void Insert(const glm::vec3 &pos, T *thing)
    {
        //std::cout<<typeid(thing).name()<<std::endl;
        mHashMap[Discretize(pos, mInvCellSize)].push_back(thing);
    }
#endif
    

    void Neighbors(const glm::vec3 &pos, NeighborList &ret) const
    {
        const glm::ivec3 ipos = Discretize(pos, mInvCellSize);
        for (const auto &offset : mOffsets)
        {
            typename HashMap::const_iterator it = mHashMap.find(offset + ipos);
            if (it != mHashMap.end())
            {
                ret.insert(ret.end(), it->second.begin(), it->second.end());
            }
        }
    }

    void Clear()
    {
        mHashMap.clear();
    }

  private:
    // "Optimized Spatial Hashing for Collision Detection of Deformable Objects"
    // Teschner, Heidelberger, et al.
    // returns a hash between 0 and 2^32-1
    struct TeschnerHash : std::unary_function<glm::ivec3, std::size_t>
    {
        std::size_t operator()(glm::ivec3 const &pos) const
        {
            const unsigned int p1 = 73856093;
            const unsigned int p2 = 19349663;
            const unsigned int p3 = 83492791;
            return size_t((pos.x * p1) ^ (pos.y * p2) ^ (pos.z * p3));
        };
    };

    // returns the indexes of the cell pos is in, assuming a cellSize grid
    // invCellSize is the inverse of the desired cell size
    static inline glm::ivec3 Discretize(const glm::vec3 &pos, const float invCellSize)
    {
        auto temp=glm::ivec3(glm::floor(pos * invCellSize));
        //std::cout<<glm::to_string(temp)<<std::endl;
        return temp;
    }

    typedef std::unordered_map<glm::ivec3, NeighborList, TeschnerHash> HashMap;
    HashMap mHashMap;

    std::vector<glm::ivec3> mOffsets;

    const float mInvCellSize;
};

typedef SpatialIndex<Particle> IndexType;
IndexType indexsp(4093, r, true);

// --------------------------------------------------------------------
void step()
{
    // UPDATE
    // This modified verlet integrator has dt = 1 and calculates the velocity
    // For later use in the simulation.
#if parallel
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)particles.size(); ++i)
    {
        //  printf("Thread rank: %d\n", omp_get_thread_num());

        // Apply the currently accumulated forces
        particles[i].pos += particles[i].force;

        // Restart the forces with gravity only. We'll add the rest later.
        particles[i].force = glm::vec2(0.0f, -::G);

        // Calculate the velocity for later.
        particles[i].vel = particles[i].pos - particles[i].pos_old;

        // If the velocity is really high, we're going to cheat and cap it.
        // This will not damp all motion. It's not physically-based at all. Just
        // a little bit of a hack.
        const float max_vel = 2.0f;
        const float vel_mag = glm::dot(particles[i].vel, particles[i].vel);
        // If the velocity is greater than the max velocity, then cut it in half.
        if (vel_mag > max_vel * max_vel)
        {
            particles[i].vel *= .5f;
        }

        // Normal verlet stuff
        particles[i].pos_old = particles[i].pos;
        particles[i].pos += particles[i].vel;

        // If the Particle is outside the bounds of the world, then
        // Make a little spring force to push it back in.
        if (particles[i].pos.x < -SIM_W)
            particles[i].force.x -= (particles[i].pos.x - -SIM_W) / 8;
        if (particles[i].pos.x > SIM_W)
            particles[i].force.x -= (particles[i].pos.x - SIM_W) / 8;
        if (particles[i].pos.y < bottom)
            particles[i].force.y -= (particles[i].pos.y - bottom) / 8;
        //if( particles[i].pos.y > SIM_W * 2 ) particles[i].force.y -= ( particles[i].pos.y - SIM_W * 2 ) / 8;

        // Handle the mouse attractor.
        // It's a simple spring based attraction to where the mouse is.
        const float attr_dist2 = glm::dot(particles[i].pos - attractor, particles[i].pos - attractor);
        const float attr_l = SIM_W / 4;
        if (attracting)
        {
            if (attr_dist2 < attr_l * attr_l)
            {
                particles[i].force -= (particles[i].pos - attractor) / 256.0f;
            }
        }

        // Reset the nessecary items.
        particles[i].rho = 0;
        particles[i].rho_near = 0;
        particles[i].neighbors.clear();
    }

    //std::cout<<"0"<<std::endl;
    // update spatial index
    indexsp.Clear();
#if idxv
    for (int i = 0; i < (int)particles.size(); ++i)
    {
        indexsp.Insert_idx(glm::vec3(particles[i].pos, 0.0f), i);
    }
#else
    for (auto &particle : particles)
    {
        //std::cout<<(int)particles.size()<<std::endl;
        //std::cout<<glm::to_string((particle).pos)<<std::endl;
        //std::cout<<typeid(particle).name()<<" ";
        indexsp.Insert(glm::vec3(particle.pos, 0.0f), &particle);
    }
#endif

    //std::cout<<"1"<<std::endl;

    // DENSITY
    // Calculate the density by basically making a weighted sum
    // of the distances of neighboring particles within the radius of support (r)
#if parallel
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)particles.size(); ++i)
    {
        particles[i].rho = 0;
        particles[i].rho_near = 0;

        // We will sum up the 'near' and 'far' densities.
        float d = 0;
        float dn = 0;

#if idxv
        IndexType::NeighborList neigh;
        //std::cout<<"10"<<std::endl;
        //neigh.reserve(64);
        indexsp.Neighbors(glm::vec3(particles[i].pos, 0.0f), neigh);
        //std::cout<<typeid(neigh[0]).name()<<" ";
        // if (neigh.size()>64)
        // std::cout<<(int)neigh.size()<<" ";
        //std::cout<<"11"<<std::endl;
        for (int j = 0; j < (int)neigh.size(); ++j)
        {
            if (neigh[j] == i)
            {
                // do not calculate an interaction for a Particle with itself!
                continue;
            }

            Particle curparticle=particles[neigh[j]];
            //std::cout<<"11"<<std::endl;

            // The vector seperating the two particles
            const glm::vec2 rij = curparticle.pos - particles[i].pos;

            // Along with the squared distance between
            const float rij_len2 = glm::dot(rij, rij);

            // If they're within the radius of support ...
            if (rij_len2 < rsq)
            {
                // Get the actual distance from the squared distance.
                float rij_len = sqrt(rij_len2);

                // And calculated the weighted distance values
                const float q = 1 - (rij_len / r);
                const float q2 = q * q;
                const float q3 = q2 * q;

                d += q2;
                dn += q3;

                // Set up the Neighbor list for faster access later.
                Neighbor n;
                n.j = neigh[j];
                n.q = q;
                particles[i].neighbors.push_back(n);
            }
        }
#else

        IndexType::NeighborList neigh;
        //neigh.reserve(64);
        indexsp.Neighbors(glm::vec3(particles[i].pos, 0.0f), neigh);
        // if (neigh.size()>64)
        // std::cout<<(int)neigh.size()<<" ";
        for (int j = 0; j < (int)neigh.size(); ++j)
        {
            if (neigh[j] == &particles[i])
            {
                // do not calculate an interaction for a Particle with itself!
                continue;
            }

            // The vector seperating the two particles
            const glm::vec2 rij = neigh[j]->pos - particles[i].pos;

            // Along with the squared distance between
            const float rij_len2 = glm::dot(rij, rij);

            // If they're within the radius of support ...
            if (rij_len2 < rsq)
            {
                // Get the actual distance from the squared distance.
                float rij_len = sqrt(rij_len2);

                // And calculated the weighted distance values
                const float q = 1 - (rij_len / r);
                const float q2 = q * q;
                const float q3 = q2 * q;

                d += q2;
                dn += q3;

                // Set up the Neighbor list for faster access later.
                Neighbor n;
                n.j = neigh[j];
                n.q = q;
                particles[i].neighbors.push_back(n);
            }
        }
#endif

        particles[i].rho += d;
        particles[i].rho_near += dn;
    }
    //std::cout<<"2"<<std::endl;

    // PRESSURE
    // Make the simple pressure calculation from the equation of state.
#if parallel
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)particles.size(); ++i)
    {
        particles[i].press = k * (particles[i].rho - rest_density);
        particles[i].press_near = k_near * particles[i].rho_near;
    }

    //printf("Finish second loop\n");

    // PRESSURE FORCE
    // We will force particles in or out from their neighbors
    // based on their difference from the rest density.
#if parallel
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)particles.size(); ++i)
    {
        // For each of the neighbors
        glm::vec2 dX(0);
        for (const Neighbor &n : particles[i].neighbors)
        {
#if idxv
            Particle curparticle=particles[n.j];
            // The vector from Particle i to Particle j
            const glm::vec2 rij = curparticle.pos - particles[i].pos;

            // calculate the force from the pressures calculated above
            const float dm = n.q * (particles[i].press + curparticle.press) + n.q * n.q * (particles[i].press_near + curparticle.press_near);

            // Get the direction of the force
            const glm::vec2 D = glm::normalize(rij) * dm;
            dX += D;
#else
            // The vector from Particle i to Particle j
            const glm::vec2 rij = (*n.j).pos - particles[i].pos;

            // calculate the force from the pressures calculated above
            const float dm = n.q * (particles[i].press + (*n.j).press) + n.q * n.q * (particles[i].press_near + (*n.j).press_near);

            // Get the direction of the force
            const glm::vec2 D = glm::normalize(rij) * dm;
            dX += D;
#endif
        }
        particles[i].force -= dX;
    }

    //printf("Finish third loop\n");

    // VISCOSITY
    // This simulation actually may look okay if you don't compute
    // the viscosity section. The effects of numerical damping and
    // surface tension will give a smooth appearance on their own.
    // Try it.
#if parallel
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)particles.size(); ++i)
    {
        // We'll let the color be determined by
        // ... x-velocity for the red component
        // ... y-velocity for the green-component
        // ... pressure for the blue component
        particles[i].r = 0.3f + (20 * fabs(particles[i].vel.x));
        particles[i].g = 0.3f + (20 * fabs(particles[i].vel.y));
        particles[i].b = 0.3f + (0.1f * particles[i].rho);

        // For each of that particles neighbors
        for (const Neighbor &n : particles[i].neighbors)
        {
#if idxv
            Particle curparticle=particles[n.j];
            const glm::vec2 rij = curparticle.pos - particles[i].pos;
            const float l = glm::length(rij);
            const float q = l / r;

            const glm::vec2 rijn = (rij / l);
            // Get the projection of the velocities onto the vector between them.
            const float u = glm::dot(particles[i].vel - curparticle.vel, rijn);
            if (u > 0)
            {
                // Calculate the viscosity impulse between the two particles
                // based on the quadratic function of projected length.
                const glm::vec2 I = (1 - q) * (curparticle.sigma * u + curparticle.beta * u * u) * rijn;

                // Apply the impulses on the current particle
                particles[i].vel -= I * 0.5f;
            }
#else
            const glm::vec2 rij = (*n.j).pos - particles[i].pos;
            const float l = glm::length(rij);
            const float q = l / r;

            const glm::vec2 rijn = (rij / l);
            // Get the projection of the velocities onto the vector between them.
            const float u = glm::dot(particles[i].vel - (*n.j).vel, rijn);
            if (u > 0)
            {
                // Calculate the viscosity impulse between the two particles
                // based on the quadratic function of projected length.
                const glm::vec2 I = (1 - q) * ((*n.j).sigma * u + (*n.j).beta * u * u) * rijn;

                // Apply the impulses on the current particle
                particles[i].vel -= I * 0.5f;
            }
#endif
        }
    }
}

// --------------------------------------------------------------------
void display(GLFWwindow *window)
{
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int w, h;
    glfwGetWindowSize(window, &w, &h);
    glViewport(0, 0, w, h);

    // create a world with dimensions x:[-SIM_W,SIM_W] and y:[0,SIM_W*2]
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const double ar = w / static_cast<double>(h);
    glOrtho(ar * -SIM_W, ar * SIM_W, 0, 2 * SIM_W, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Draw Fluid Particles
    glPointSize(r * 2);
    glVertexPointer(2, GL_FLOAT, sizeof(Particle), &particles[0].pos);
    glColorPointer(3, GL_FLOAT, sizeof(Particle), &particles[0].r);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, particles.size());
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

// --------------------------------------------------------------------
unsigned int stepsPerFrame = 1;
void keyboard(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        return;
    }

    const float radius = SIM_W / 8;
    switch (key)
    {
    case GLFW_KEY_ESCAPE:
    case GLFW_KEY_Q:
        glfwSetWindowShouldClose(window, 1);
        break;
    case GLFW_KEY_SPACE:
        // add some particles.
        for (float y = SIM_W * 2 - radius; y <= SIM_W * 2 + radius; y += r * .5f)
        {
            for (float x = -radius; x <= radius; x += r * .5f)
            {
                Particle p;
                p.pos = p.pos_old = glm::vec2(x, y) + glm::vec2(rand01(), rand01());
                p.force = glm::vec2(0, 0);
                p.sigma = 3.f;
                p.beta = 4.f;

                const glm::vec2 temp(p.pos - glm::vec2(0, SIM_W * 2));
                if (glm::dot(temp, temp) < radius * radius)
                {
                    particles.push_back(p);
                }
            }
        }
        break;
    case GLFW_KEY_MINUS:
    case GLFW_KEY_KP_SUBTRACT:
        if (stepsPerFrame > 1)
            stepsPerFrame--;
        break;
    case GLFW_KEY_EQUAL:
    case GLFW_KEY_KP_ADD:
        stepsPerFrame++;
        break;
    }
}

// --------------------------------------------------------------------
void motion(GLFWwindow *window, double xpos, double ypos)
{
    // This simply updates the location of the mouse attractor.
    int window_w, window_h;
    glfwGetWindowSize(window, &window_w, &window_h);
    float relx = (float)(xpos - window_w / 2) / window_w;
    float rely = -(float)(ypos - window_h) / window_h;
    glm::vec2 mouse = glm::vec2(relx * SIM_W * 2, rely * SIM_W * 2);
    attractor = mouse;
}

// --------------------------------------------------------------------
void mouse(GLFWwindow *window, int button, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        attracting = true;
    }
    else
    {
        attracting = false;
        attractor = glm::vec2(SIM_W * 99, SIM_W * 99);
    }
}

// --------------------------------------------------------------------
int main(int argc, char **argv)
{
#if 0
    const int steps = 3000;
    std::cout << "--------------------------------" << std::endl;
    std::cout  << "Number of steps: " << steps << std::endl;
    for( unsigned int size = 10; size <= 13; ++size )
    {
        const unsigned int count = ( 1 << size );
        std::cout << "Number of particles: " << count << std::endl;

        init( count );

        const auto beg = std::chrono::high_resolution_clock::now();
        for( unsigned int i = 0; i < steps; ++i )
        {
            step();
        }
        const auto end = std::chrono::high_resolution_clock::now();

        const auto duration( end - beg );
        std::cout << "Elapsed time: " << std::chrono::duration_cast< std::chrono::milliseconds >( duration ).count() << " milliseconds" << std::endl;
        std::cout << "Microseconds per step: " << std::chrono::duration_cast< std::chrono::microseconds >( duration ).count() / (double)steps << std::endl;
        std::cout << std::endl;
    }

    return 0;
#else
    // init(2048);
    init(8192);

    glfwInit();

    GLFWwindow *window = glfwCreateWindow(512, 512, "SPH", NULL, NULL);

    glfwMakeContextCurrent(window);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSwapInterval(1);

    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, motion);
    glfwSetMouseButtonCallback(window, mouse);

    int framecount = 0;
    clock_t now = std::clock(), prev = std::clock();

    double total = 0;
    double ttotal = 0;
    // double start = CycleTimer::currentSeconds();
    // while (!glfwWindowShouldClose(window))  
    while (framecount < 500+ 1)
    {

        // display( window );
        if (framecount % 50 == 0)
        {
            // start=std::chrono::high_resolution_clock::now();
            // now = std::clock();
            // std::cout << "Duration: " << (now - prev) / (double)3300000000 << std::endl;
            // prev = now;

            std::cout << "Duration: " << total << std::endl;
            total = 0;
        }

        glfwPollEvents();

        double start = CycleTimer::currentSeconds();
        for (size_t i = 0; i < stepsPerFrame; ++i)
        {
            framecount++;
            step();
        }
        double diff = CycleTimer::currentSeconds() - start;
        total += diff;
        ttotal += diff;

        glfwSwapBuffers(window);
    }
    // ttotal = CycleTimer::currentSeconds() - start;
    std::cout << "Average Duration: " << ttotal / framecount << std::endl;

    glfwTerminate();
    return 0;
#endif
}