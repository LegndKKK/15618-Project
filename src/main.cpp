#include "particle.h"


// --------------------------------------------------------------------
// Between [0,1]
float rand01()
{
    return (float)rand() * (1.f / RAND_MAX);
}
// Our collection of particles
// std::vector<Particle> particles;
Particle *particles;
// 2-d array of particles' neighbors
short **neighborIndex;  // (N, NEIGHBOR_NUM)
float **neighborDist; // (N, NEIGHBOR_NUM)
short *neighborNum;     // (N, )

// --------------------------------------------------------------------
void init(const unsigned int N)
{
    // for CUDA use
    neighborIndex = (short **)malloc(N * sizeof(short *));
    neighborDist = (float **)malloc(N * sizeof(float *));
    neighborNum = (short *)calloc(N, sizeof(short));
    particles = (Particle *)malloc(N * sizeof(Particle));

    // Initialize particles
    // We will make a block of particles with a total width of 1/4 of the screen.
    float w = SIM_W / 4;
    int i = 0;
    for (float y = bottom + 1; y <= 100000; y += r * 0.5f)
    {
        for (float x = -w; x <= w; x += r * 0.5f)
        {
            if (i == N)
            {
                break;
            }

            Particle p;
            p.pos = glm::vec2(x, y);
            p.pos_old = p.pos + 0.001f * glm::vec2(rand01(), rand01());
            p.force = glm::vec2(0, 0);
            p.id = i;

            particles[i] = p;

            neighborIndex[i] = (short *)calloc(NEIGHBOR_NUM, sizeof(short));
            neighborDist[i] = (float *)calloc(NEIGHBOR_NUM, sizeof(float));
            i++;
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
    typedef std::vector<T *> NeighborList;

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

    void Insert(const glm::vec3 &pos, T *thing)
    {
        mHashMap[Discretize(pos, mInvCellSize)].push_back(thing);
    }

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
        return glm::ivec3(glm::floor(pos * invCellSize));
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
#if OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < PARTICLE_NUM; ++i)
    {
        //  printf("Thread rank: %d\n", omp_get_thread_num());
        // std::cout<< sizeof(particles[i]) << ", "<< (&particles[i] - &particles[0]) / sizeof(particles[i]) <<", "<< i <<std::endl;

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
    memset(neighborNum, 0, PARTICLE_NUM * sizeof(short));

    // update spatial index
    indexsp.Clear();
    // for (auto &particle : particles)
    // {
    //     indexsp.Insert(glm::vec3(particle.pos, 0.0f), &particle);
    // }
    for (int t = 0; t < PARTICLE_NUM; t++)
    {
        indexsp.Insert(glm::vec3(particles[t].pos, 0.0f), &particles[t]);
    }

// DENSITY
// Calculate the density by basically making a weighted sum
// of the distances of neighboring particles within the radius of support (r)
#if OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < PARTICLE_NUM; ++i)
    {
        particles[i].rho = 0;
        particles[i].rho_near = 0;

        // We will sum up the 'near' and 'far' densities.
        float d = 0;
        float dn = 0;

        IndexType::NeighborList neigh;
        neigh.reserve(64);
        indexsp.Neighbors(glm::vec3(particles[i].pos, 0.0f), neigh);
        // if ((int)neigh.size() > 100) {
        //     std::cout<< neigh.size() << std::endl;
        // }

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
                n.q2 = q2;
                particles[i].neighbors.push_back(n);

                // for CUDA use
                // std::cout<<i<<std::endl;
                neighborIndex[i][neighborNum[i]] = (short)n.j->id;
                neighborDist[i][neighborNum[i]] = n.q;
                neighborNum[i]++;
            }
        }
        // if (particles[i].neighbors.size() > 200)
        // {
        //     std::cout << neighborNum[i] << ", " << particles[i].neighbors.size() << std::endl;
        // }

        particles[i].rho += d;
        particles[i].rho_near += dn;
    }

// PRESSURE
// Make the simple pressure calculation from the equation of state.
#if CUDA
    Surprise_CUDA(PARTICLE_NUM, particles, (short *)neighborNum, (short *)neighborIndex, (float *)neighborDist);
#else
#if OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)PARTICLE_NUM; ++i)
    {
        particles[i].press = k * (particles[i].rho - rest_density);
        particles[i].press_near = k_near * particles[i].rho_near;
    }

// printf("---------\n");
// for (int i = 0; i < 5; ++i)
//     {
//         printf("%f\n", particles[i].press);
//     }

// PRESSURE FORCE
// We will force particles in or out from their neighbors
// based on their difference from the rest density.
#if OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < PARTICLE_NUM; ++i)
    {
        // For each of the neighbors
        glm::vec2 dX(0);
        // for (const Neighbor &n : particles[i].neighbors)
        // {
        //     // The vector from Particle i to Particle j
        //     const glm::vec2 rij = (*n.j).pos - particles[i].pos;

        //     // calculate the force from the pressures calculated above
        //     const float dm = n.q * (particles[i].press + (*n.j).press) + n.q2 * (particles[i].press_near + (*n.j).press_near);

        //     // Get the direction of the force
        //     const glm::vec2 D = glm::normalize(rij) * dm;
        //     dX += D;
        // }

        for (int j = 0; j < neighborNum[i]; j++)
        {
            // std::cout<<"A1"<<std::endl;
            Particle p = particles[neighborIndex[i][j]];
            // std::cout<<"A2"<<std::endl;
            // if (i > 2048 || j > 300) std::cout<<i<<", "<<j<<std::endl;
            float dist_q = neighborDist[i][j];
            // std::cout<<"A3"<<std::endl;
            // The vector from Particle i to Particle j
            const glm::vec2 rij = p.pos - particles[i].pos;

            // calculate the force from the pressures calculated above
            const float dm = dist_q * (particles[i].press + p.press) + (dist_q * dist_q) * (particles[i].press_near + p.press_near);

            // Get the direction of the force
            const glm::vec2 D = glm::normalize(rij) * dm;
            dX += D;
        }

        particles[i].force -= dX;
    }

// std::cout<<"B"<<std::endl;
// VISCOSITY
// This simulation actually may look okay if you don't compute
// the viscosity section. The effects of numerical damping and
// surface tension will give a smooth appearance on their own.
// Try it.
#if OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < PARTICLE_NUM; ++i)
    {
        // We'll let the color be determined by
        // ... x-velocity for the red component
        // ... y-velocity for the green-component
        // ... pressure for the blue component
        particles[i].r = 0.3f + (20 * fabs(particles[i].vel.x));
        particles[i].g = 0.3f + (20 * fabs(particles[i].vel.y));
        particles[i].b = 0.3f + (0.1f * particles[i].rho);

        // For each of that particles neighbors
        // for (const Neighbor &n : particles[i].neighbors)
        // {
        //     const glm::vec2 rij = (*n.j).pos - particles[i].pos;
        //     const float l = glm::length(rij);
        //     const float q = l / r;

        //     const glm::vec2 rijn = (rij / l);
        //     // Get the projection of the velocities onto the vector between them.
        //     const float u = glm::dot(particles[i].vel - (*n.j).vel, rijn);
        //     if (u > 0)
        //     {
        //         // Calculate the viscosity impulse between the two particles
        //         // based on the quadratic function of projected length.
        //         const glm::vec2 I = (1 - q) * ((*n.j).sigma * u + (*n.j).beta * u * u) * rijn;

        //         // Apply the impulses on the current particle
        //         particles[i].vel -= I * 0.5f;
        //     }
        // }

        for (int j = 0; j < neighborNum[i]; j++)
        {
            Particle p = particles[neighborIndex[i][j]];

            const glm::vec2 rij = p.pos - particles[i].pos;
            const float l = glm::length(rij);
            const float q = l / r;

            const glm::vec2 rijn = (rij / l);
            // Get the projection of the velocities onto the vector between them.
            const float u = glm::dot(particles[i].vel - p.vel, rijn);
            if (u > 0)
            {
                // Calculate the viscosity impulse between the two particles
                // based on the quadratic function of projected length.
                const glm::vec2 I = (1 - q) * (sigma * u + beta * u * u) * rijn;

                // Apply the impulses on the current particle
                particles[i].vel -= I * 0.5f;
            }
        }
    }
#endif
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
    glDrawArrays(GL_POINTS, 0, PARTICLE_NUM);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

// --------------------------------------------------------------------
unsigned int stepsPerFrame = 1;
// void keyboard(GLFWwindow *window, int key, int scancode, int action, int mods)
// {
//     if (action == GLFW_PRESS)
//     {
//         return;
//     }

//     const float radius = SIM_W / 8;
//     switch (key)
//     {
//     case GLFW_KEY_ESCAPE:
//     case GLFW_KEY_Q:
//         glfwSetWindowShouldClose(window, 1);
//         break;
//     case GLFW_KEY_SPACE:
//         // add some particles.
//         for (float y = SIM_W * 2 - radius; y <= SIM_W * 2 + radius; y += r * .5f)
//         {
//             for (float x = -radius; x <= radius; x += r * .5f)
//             {
//                 Particle p;
//                 p.pos = p.pos_old = glm::vec2(x, y) + glm::vec2(rand01(), rand01());
//                 p.force = glm::vec2(0, 0);
//                 p.sigma = 3.f;
//                 p.beta = 4.f;

//                 const glm::vec2 temp(p.pos - glm::vec2(0, SIM_W * 2));
//                 if (glm::dot(temp, temp) < radius * radius)
//                 {
//                     // particles.push_back(p);
//                 }
//             }
//         }
//         break;
//     case GLFW_KEY_MINUS:
//     case GLFW_KEY_KP_SUBTRACT:
//         if (stepsPerFrame > 1)
//             stepsPerFrame--;
//         break;
//     case GLFW_KEY_EQUAL:
//     case GLFW_KEY_KP_ADD:
//         stepsPerFrame++;
//         break;
//     }
// }

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
    std::cout<<"PARTICLE_NUM :"<<(PARTICLE_NUM)<<std::endl;
    std::cout<<"NEIGHBOR_NUM :"<<(NEIGHBOR_NUM)<<std::endl;
    std::cout<<"OMP :"<<OMP<<std::endl;
    std::cout<<"CUDA :"<<CUDA<<std::endl;
    //printf("%d\n",num_of_threads);
    init(PARTICLE_NUM);

    glfwInit();

    GLFWwindow *window = glfwCreateWindow(512, 512, "SPH", NULL, NULL);

    glfwMakeContextCurrent(window);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSwapInterval(1);

    // glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, motion);
    glfwSetMouseButtonCallback(window, mouse);

    int framecount = 0;

    double total = 0;
    double ttotal = 0;
    double start = CycleTimer::currentSeconds();
    // while (!glfwWindowShouldClose(window))
    while (framecount < 10 * 50)
    {

        // display( window );
        if (framecount % 50 == 0)
        {
            std::cout << "Framecount: " << framecount << std::endl;
            std::cout << "Duration: " << CycleTimer::currentSeconds() - start << std::endl;
            total = 0;
        }

        // glfwPollEvents();

        // double start = CycleTimer::currentSeconds();
        for (size_t i = 0; i < stepsPerFrame; ++i)
        {
            step();
        }
        framecount += stepsPerFrame;
        // double diff = CycleTimer::currentSeconds() - start;
        // total += diff;
        // ttotal += diff;d

        // glfwSwapBuffers(window);
    }
    ttotal = CycleTimer::currentSeconds() - start;
    std::cout << "Average Duration: " << ttotal / framecount << std::endl;

    glfwTerminate();
    return 0;
}
