#include "particle.h"

#define PPP cudaDeviceParticles[cudaDeviceNeighborIndex[i*NEIGHBOR_NUM+j]]


__global__ void kernel_pressure(Particle *cudaDeviceParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%d ",i);
    if (i < PARTICLE_NUM) {
        cudaDeviceParticles[i].press = 0.002f * (cudaDeviceParticles[i].rho -3.f);
        cudaDeviceParticles[i].press_near = 0.02f* cudaDeviceParticles[i].rho_near;
    }
}

__global__ void kernel_pressure_force(Particle *cudaDeviceParticles, int *cudaDeviceNeighborNum, int *cudaDeviceNeighborIndex, float *cudaDeviceNeighborDist) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%d ",i);
    if (i < PARTICLE_NUM) {
        glm::vec2 dX(0);
        for (int j = 0; j < cudaDeviceNeighborNum[i]; j++)
        {
            // Particle p = cudaDeviceParticles[cudaDeviceNeighborIndex[i*NEIGHBOR_NUM+j]];
            float dist_q = cudaDeviceNeighborDist[i*NEIGHBOR_NUM+j];
            // The vector from Particle i to Particle j
            const glm::vec2 rij = PPP.pos - cudaDeviceParticles[i].pos;

            // calculate the force from the pressures calculated above
            const float dm = dist_q * (cudaDeviceParticles[i].press + PPP.press) + (dist_q * dist_q) * (cudaDeviceParticles[i].press_near + PPP.press_near);

            // Get the direction of the force
            const glm::vec2 D = glm::normalize(rij) * dm;
            dX += D;
        }
        cudaDeviceParticles[i].force -= dX;
    }
}

__global__ void kernel_viscosity(Particle *cudaDeviceParticles, int *cudaDeviceNeighborNum, int *cudaDeviceNeighborIndex, float *cudaDeviceNeighborDist) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < PARTICLE_NUM) 
    {
        cudaDeviceParticles[i].r = 0.3f + (20 * fabs(cudaDeviceParticles[i].vel.x));
        cudaDeviceParticles[i].g = 0.3f + (20 * fabs(cudaDeviceParticles[i].vel.y));
        cudaDeviceParticles[i].b = 0.3f + (0.1f * cudaDeviceParticles[i].rho);
        for (int j = 0; j < cudaDeviceNeighborNum[i]; j++)
        {
            const glm::vec2 rij = PPP.pos - cudaDeviceParticles[i].pos;
            const float l = glm::length(rij);
            const float q = l / 2.5f;

            const glm::vec2 rijn = (rij / l);
            // Get the projection of the velocities onto the vector between them.
            const float u = glm::dot(cudaDeviceParticles[i].vel - PPP.vel, rijn);
            if (u > 0)
            {
                // Calculate the viscosity impulse between the two particles
                // based on the quadratic function of projected length.
                const glm::vec2 I = (1 - q) * (3.f * u + 4.f* u * u) * rijn;

                // Apply the impulses on the current particle
                cudaDeviceParticles[i].vel -= I * 0.5f;
            }
        }
    }
}

void Surprise_CUDA(long N, Particle *particles, int *neighborNum, int *neighborIndex, float *neighborDist)
{
    // globalConstants cuconstantParams;
    // cudaMemcpyToSymbol(cuconstantParams,(void *)&constantParams,sizeof(globalConstants));

    Particle *cudaDeviceParticle=NULL;
    int *cudaDeviceNeighborNum = NULL;
    int *cudaDeviceNeighborIndex=NULL;
    float *cudaDeviceNeighborDist = NULL;
    
    cudaMalloc(&cudaDeviceParticle, sizeof(Particle) * N);
    cudaMalloc(&cudaDeviceNeighborNum, sizeof(int)* N );
    cudaMalloc(&cudaDeviceNeighborIndex, sizeof(int)* N * NEIGHBOR_NUM);
    cudaMalloc(&cudaDeviceNeighborDist, sizeof(float)* N * NEIGHBOR_NUM);

    cudaMemcpy(cudaDeviceParticle, particles, sizeof(Particle) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceNeighborNum, neighborNum, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceNeighborIndex, neighborIndex, sizeof(int) * N * NEIGHBOR_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceNeighborDist, neighborDist, sizeof(float) * N * NEIGHBOR_NUM, cudaMemcpyHostToDevice);

    
    dim3 blockDim(1024, 1, 1);
    dim3 gridDim((PARTICLE_NUM+blockDim.x-1)/blockDim.x,1,1);
    kernel_pressure<<<gridDim,blockDim>>>(cudaDeviceParticle);
    cudaDeviceSynchronize();


    kernel_pressure_force<<<gridDim,blockDim>>>(cudaDeviceParticle, cudaDeviceNeighborNum, cudaDeviceNeighborIndex, cudaDeviceNeighborDist);
    cudaDeviceSynchronize();

    kernel_viscosity<<<gridDim,blockDim>>>(cudaDeviceParticle, cudaDeviceNeighborNum, cudaDeviceNeighborIndex, cudaDeviceNeighborDist);
    cudaDeviceSynchronize();

    cudaMemcpy(particles, cudaDeviceParticle,sizeof(Particle) * N, cudaMemcpyDeviceToHost);
    return;
}
