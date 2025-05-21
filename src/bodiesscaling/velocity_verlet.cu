#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

// different output file for specific properties of the system
const std::string kFilename = "out100000.xyz";
const std::string kEnergyFilename = "out100000.energy";
const std::string kMomentumFilename = "out100000.momentum";

// setup specific parameters
const int kNumBodies = 100000;
__device__ const float kDeltaT = 0.001;
const int kNumTimesteps = 900;
const int kNumTimestepsSnapshot = 100000;

// physical constant
__device__ const float kGravitationalConstant = 6.67430e-11;
__device__ const float kEpsilon = 0;

// cuda block and grid size
const int kBlockDim = 16;
constexpr int kGridDim = ((kNumBodies-1)/kBlockDim)+1;


// use a 3d vector in cuda
struct Vec3 {
    float x;
    float y;
    float z;

    float operator[](int idx) const {
        if (idx == 0) {
            return x;
        }
        if (idx == 1) {
            return y;
        }
        return z;
    }

    // operator overloads to be able to do component wise operations
    __host__ __device__ Vec3 operator+(const Vec3& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }
    __host__ __device__ Vec3 operator-(const Vec3& rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }
    __host__ __device__ Vec3 operator*(const Vec3& rhs) const {
        return {x * rhs.x, y * rhs.y, z * rhs.z};
    }
    __host__ __device__ Vec3 operator*(const float& rhs) const {
        return {x * rhs, y * rhs, z * rhs};
    }
    __host__ __device__ Vec3 operator/(const Vec3& rhs) const {
        return {x / rhs.x, y / rhs.y, z / rhs.z};
    }
    __host__ __device__ Vec3 operator/(const float& rhs) const {
        return {x / rhs, y / rhs, z / rhs};
    }
    __host__ __device__ bool operator==(const Vec3& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};
// calculate the norm of a 3d vector
__host__ __device__ float GetNorm(const Vec3& vec) {
    return std::sqrt((vec.x*vec.x) + (vec.y*vec.y) + (vec.z*vec.z));
}
// norm with a epsilon component to avoid devision by zero if epsilon is not 0
__host__ __device__ float GetNormEpsilon(const Vec3& vec) {
    return std::sqrt((vec.x*vec.x) + (vec.y*vec.y) + (vec.z*vec.z) + (kEpsilon*kEpsilon));
}


// representation of a body
struct Body {
    Vec3 pos;    // position
    Vec3 vel;    // velocity
    Vec3 acc;    // acceleration
    float mass;  // mass
    float potential_energy = 0;
    float kinetic_energy = 0;
};


// update the position on the GPU
__global__ void UpdatePosition(Body* bodies) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= kNumBodies) {
        return;
    }
    bodies[index].pos = bodies[index].pos + bodies[index].vel * kDeltaT;
}


// update the velocity on the GPU
__global__ void UpdateVelocityHalf(Body* bodies) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= kNumBodies) {
        return;
    }
    bodies[index].vel = bodies[index].vel + bodies[index].acc * kDeltaT / 2;
}


// calculate the gravitational force
__device__ Vec3 GravitationalForce(const Body& first, const Body& second) {
    float norm = GetNormEpsilon(second.pos - first.pos);
    Vec3 force = (second.pos - first.pos) * kGravitationalConstant * first.mass * second.mass / (norm * norm * norm);
    return force;
}


// derive the acceleration at a specific moment in time
__global__ void DeriveAcc(Body* bodies) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= kNumBodies) {
        return;
    }
    // sum up all forces of the different other bodies
    bodies[index].acc = {0, 0, 0};
    for (int i = 0; i < kNumBodies; i++) {
        if (i == index) {
            continue;
        }
        bodies[index].acc = bodies[index].acc + GravitationalForce(bodies[index], bodies[i]) / bodies[index].mass;
    }
}


// calculate the potential energy of two bodies
__device__ float GetPotential(const Body& first, const Body& second) {
    return -kGravitationalConstant * first.mass * second.mass / GetNormEpsilon(second.pos - first.pos);
}


// calculate the potential energy of all bodies
__global__ void CalculatePotential(Body* bodies) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= kNumBodies) {
        return;
    }
    bodies[index].potential_energy = 0;
    for (int i = 0; i < kNumBodies; i++) {
        if (i == index) {
            continue;
        }
        bodies[index].potential_energy = bodies[index].potential_energy + GetPotential(bodies[index], bodies[i]);
    }
}


// calculate the kinetic energy of all bodies
__global__ void CalculateKinetic(Body* bodies) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= kNumBodies) {
        return;
    }
    bodies[index].kinetic_energy = 1. / 2 * bodies[index].mass * GetNorm(bodies[index].vel) * GetNorm(bodies[index].vel);
}


int main() {
    // clear all old output files and open them
    {
        std::ofstream outfile;
        outfile.open(kFilename);
        outfile << kNumBodies << " " << kNumTimesteps << " " << kNumTimestepsSnapshot << "\n";
        outfile.close();
        std::ofstream potential_file;
        std::ofstream momentum_file;
        potential_file.open(kEnergyFilename);
        momentum_file.open(kMomentumFilename);
        potential_file.close();
        momentum_file.close();
    }
    std::ofstream outfile(kFilename, std::ios_base::app);
    std::ofstream potential_file(kEnergyFilename, std::ios_base::app);
    std::ofstream momentum_file(kMomentumFilename, std::ios_base::app);

    // create the bodies on the CPU
    Body* bodies = (Body*)malloc(sizeof(Body) * kNumBodies);
    // allocate the bodies on the device GPU
    Body* d_bodies;
    cudaMalloc((void**)&d_bodies, sizeof(Body) * kNumBodies);

    // set the position randomly in 3d space
    std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution_x(-50, 50);
    std::uniform_real_distribution<float> distribution_y(-50, 50);
    std::uniform_real_distribution<float> distribution_z(-2, 2); // top
    std::uniform_real_distribution<float> distribution_mass(1e1, 1e1);
    std::uniform_real_distribution<float> distributaion_vel_x(-0.1, 0.1);
    std::uniform_real_distribution<float> distributaion_vel_y(-0.1, 0.1);

    for (int i = 0; i < kNumBodies; i++) {
        Vec3 rand_pos = {distribution_x(gen), distribution_y(gen), distribution_z(gen)};
        bodies[i] = {rand_pos, {distributaion_vel_x(gen), distributaion_vel_y(gen), 0}, {0, 0, 0}, distribution_mass(gen)};
    }

    // copy the bodies from CPU to GPU
    cudaMemcpy(d_bodies, bodies, sizeof(Body) * kNumBodies, cudaMemcpyHostToDevice);

    // get the starting values
    float start_potential_energy = 0;
    float start_kinetic_energy = 0;
    {
        CalculatePotential<<<kGridDim, kBlockDim>>>(d_bodies);
        CalculateKinetic<<<kGridDim, kBlockDim>>>(d_bodies);
        cudaMemcpy(bodies, d_bodies, sizeof(Body) * kNumBodies, cudaMemcpyDeviceToHost);
        for (int i = 0; i < kNumBodies; i++) {
            start_potential_energy += bodies[i].potential_energy / 2;
            start_kinetic_energy += bodies[i].kinetic_energy;
        }
    }

    // simulate timesteps
    for (int timestep = 0; timestep < kNumTimesteps; timestep++) {
        // velocity verlet algorithm run on the GPU
        UpdateVelocityHalf<<<kGridDim, kBlockDim>>>(d_bodies);
        UpdatePosition<<<kGridDim, kBlockDim>>>(d_bodies);
        DeriveAcc<<<kGridDim, kBlockDim>>>(d_bodies);
        UpdateVelocityHalf<<<kGridDim, kBlockDim>>>(d_bodies);
        cudaDeviceSynchronize();
        std::cout << timestep << std::endl;

        if (timestep % kNumTimestepsSnapshot == kNumTimestepsSnapshot-1) {
            // save every kNumTimestepsSnapshot frame and calculate the energy state
            CalculatePotential<<<kGridDim, kBlockDim>>>(d_bodies);
            CalculateKinetic<<<kGridDim, kBlockDim>>>(d_bodies);
            cudaMemcpy(bodies, d_bodies, sizeof(Body) * kNumBodies, cudaMemcpyDeviceToHost);
            float potential_energy = 0;
            float kinetic_energy = 0;
            Vec3 total_momentum = {0, 0, 0};
            for (int i = 0; i < kNumBodies; i++) {
                potential_energy += bodies[i].potential_energy / 2;
                kinetic_energy += bodies[i].kinetic_energy;
                total_momentum = total_momentum + bodies[i].vel * bodies[i].mass;
            }

            // write the energy state to file
            std::cout << timestep << std::endl;
            potential_file << timestep * kDeltaT << " " << potential_energy-start_potential_energy << " " << kinetic_energy-start_kinetic_energy << " "<< potential_energy-start_potential_energy + kinetic_energy-start_kinetic_energy << std::endl;
            momentum_file << timestep * kDeltaT << " " << total_momentum.x << " " << total_momentum.y << " " << total_momentum.z << " " << GetNorm(total_momentum) << std::endl;
            for (int i = 0; i < kNumBodies; i++) {
                outfile << bodies[i].pos.x << " " << bodies[i].pos.y << " " << bodies[i].pos.z << "\n";
            }
            outfile << "\n";
        }
    }
}
