#include <cuda.h>
#include <fstream>
#include <iostream>
#include <string>

// different output file for specific properties of the system
const std::string kFilename = "out.xyz";
const std::string kEnergyFilename = "out.energy";
const std::string kMomentumFilename = "out.momentum";

// setup specific parameters
const int kNumBodies = 3;
__device__ const double kDeltaT = 1; // s
const int kNumTimesteps = 31556952+ 10000;
const int kNumTimestepsSnapshot = 100;

// physical constant
__device__ const double kGravitationalConstant = 6.67430e-11;
__device__ const double kEpsilon = 0;

// cuda block and grid size
const int kBlockDim = 32;
constexpr int kGridDim = ((kNumBodies-1)/kBlockDim)+1;


// use a 3d vector in cuda
struct Vec3 {
    double x;
    double y;
    double z;

    double operator[](int idx) const {
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
    __host__ __device__ Vec3 operator*(const double& rhs) const {
        return {x * rhs, y * rhs, z * rhs};
    }
    __host__ __device__ Vec3 operator/(const Vec3& rhs) const {
        return {x / rhs.x, y / rhs.y, z / rhs.z};
    }
    __host__ __device__ Vec3 operator/(const double& rhs) const {
        return {x / rhs, y / rhs, z / rhs};
    }
    __host__ __device__ bool operator==(const Vec3& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};
// calculate the norm of a 3d vector
__host__ __device__ double GetNorm(const Vec3& vec) {
    return std::sqrt((vec.x*vec.x) + (vec.y*vec.y) + (vec.z*vec.z));
}
// norm with a epsilon component to avoid devision by zero if epsilon is not 0
__host__ __device__ double GetNormEpsilon(const Vec3& vec) {
    return std::sqrt((vec.x*vec.x) + (vec.y*vec.y) + (vec.z*vec.z) + (kEpsilon*kEpsilon));
}


// representation of a body
struct Body {
    Vec3 pos;    // position
    Vec3 vel;    // velocity
    Vec3 acc;    // acceleration
    double mass;  // mass
    double potential_energy = 0;
    double kinetic_energy = 0;
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
    double norm = GetNormEpsilon(second.pos - first.pos);
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
__device__ double GetPotential(const Body& first, const Body& second) {
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

    bodies[0] = {{0.000000e+00, 0.000000e+00, 0.000000e+00}, {0.000000e+00, 0.000000e+00, 0.000000e+00}, {0, 0, 0}, 1.9885e+30};
    bodies[1] = {{-2.649903e+10, 1.446973e+11, -6.111494e+05}, {-2.979426e+04, -5.469295e+03, 1.817837e-01}, {0, 0, 0}, 5.9722e+24};
    bodies[2] = {{-2.679064e+10, 1.444223e+11, 3.566005e+07}, {-2.915073e+04, -6.200279e+03, -1.132468e+01}, {0, 0, 0}, 7.342e+22};

    // copy the bodies from CPU to GPU
    cudaMemcpy(d_bodies, bodies, sizeof(Body) * kNumBodies, cudaMemcpyHostToDevice);

    // get the starting values
    double start_potential_energy = 0;
    double start_kinetic_energy = 0;
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

        if (timestep % kNumTimestepsSnapshot == kNumTimestepsSnapshot-1) {
            // save every kNumTimestepsSnapshot frame and calculate the energy state
            CalculatePotential<<<kGridDim, kBlockDim>>>(d_bodies);
            CalculateKinetic<<<kGridDim, kBlockDim>>>(d_bodies);
            cudaMemcpy(bodies, d_bodies, sizeof(Body) * kNumBodies, cudaMemcpyDeviceToHost);
            double potential_energy = 0;
            double kinetic_energy = 0;
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
