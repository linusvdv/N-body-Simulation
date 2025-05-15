#include <cuda.h>
#include <string>
#include <fstream>
#include <random>
#include <iostream>

#define folat double

const std::string kFilename = "RKout.xyz";
const std::string kPotentialFilename = "RKout.energy";
const std::string kMomentumFilename = "RKout.momentum";
const int kBlockDim = 64;
const int kNumParticles = 3;
__device__ const float kDeltaT = 0.0003;
__device__ const float kGravitationalConstant = 6.67430e-11;
const int kNumTimesteps = 2700000;
const int kNumTimestepsSnapshot = 100;
__device__ const float kDDistance = 0;
constexpr int kGridDim = ((kNumParticles-1)/kBlockDim)+1;


struct Vec3 {
    float x = 0;
    float y = 0;
    float z = 0;

    float operator[](int idx) const {
        if (idx == 0) {
            return x;
        }
        if (idx == 1) {
            return y;
        }
        return z;
    }

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


struct State {
    Vec3 pos;
    Vec3 vel;

    __host__ __device__ State operator+(const State& rhs) const {
        return {pos + rhs.pos, vel + rhs.vel};
    }
    __host__ __device__ State operator*(const float& rhs) const {
        return {pos * rhs, vel * rhs};
    }
};


struct Particle {
    State state;

    float mass;
    float radius;
    float potential_energy = 0;
    float kinetic_energy = 0;

    State ks[4 + 1] = {};  // runge kutta 4 plus 0th index
};


__host__ __device__ float GetChangedNorm(const Vec3& vec) {
    return std::sqrt((vec.x*vec.x) + (vec.y*vec.y) + (vec.z*vec.z) + (kDDistance*kDDistance));
}


__host__ __device__ float GetNorm(const Vec3& vec) {
    return std::sqrt((vec.x*vec.x) + (vec.y*vec.y) + (vec.z*vec.z));
}


__device__ Vec3 GetUnitVector(const Vec3& vec) {
    return vec / GetNorm(vec);
}


__device__ Vec3 GravitationalAcceleration(const State& first, const State& second, const float& other_mass) {
    float norm = GetChangedNorm(second.pos - first.pos);
    Vec3 force = (second.pos - first.pos) * kGravitationalConstant * other_mass / (norm * norm * norm);
    return force;
}


__device__ float GetPotential(const Particle& first, const Particle& second) {
    return -kGravitationalConstant * first.mass * second.mass / GetChangedNorm(second.state.pos - first.state.pos);
}


__global__ void UpdatePositionVelocity(Particle* particles) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= kNumParticles) {
        return;
    }
    particles[index].state = particles[index].state + (particles[index].ks[1]
                                                     + particles[index].ks[2] * 2
                                                     + particles[index].ks[3] * 2
                                                     + particles[index].ks[4]) * (kDeltaT / 6);
}


template<int k>
__global__ void DeriveKS(Particle* particles) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= kNumParticles) {
        return;
    }
    float devising_factor[] = {1, 2, 2, 1};
    Vec3 acc = {0, 0, 0};
    for (int i = 0; i < kNumParticles; i++) {
        if (i == index) {
            continue;
        }
        acc = acc + GravitationalAcceleration((particles[index].state + particles[index].ks[k-1] * (kDeltaT / devising_factor[k-1])),
                                              (particles[i].state + particles[i].ks[k-1] * (kDeltaT / devising_factor[k-1])),
                                              particles[i].mass);
    }
    particles[index].ks[k] = {particles[index].state.vel, acc};
}


__global__ void CalculatePotential(Particle* particles) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= kNumParticles) {
        return;
    }
    particles[index].potential_energy = 0;
    for (int i = 0; i < kNumParticles; i++) {
        if (i == index) {
            continue;
        }
        particles[index].potential_energy = particles[index].potential_energy + GetPotential(particles[index], particles[i]);
    }
}


__global__ void CalculateKinetic(Particle* particles) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= kNumParticles) {
        return;
    }
    particles[index].kinetic_energy = 1. / 2 * particles[index].mass * GetNorm(particles[index].state.vel) * GetNorm(particles[index].state.vel);
}


int main() {
    {
        std::ofstream outfile;
        outfile.open(kFilename);
        outfile << kNumParticles << " " << kNumTimesteps << " " << kNumTimestepsSnapshot << "\n";
        outfile.close();
        std::ofstream potential_file;
        std::ofstream momentum_file;
        potential_file.open(kPotentialFilename);
        momentum_file.open(kMomentumFilename);
        potential_file.close();
        momentum_file.close();
    }
    std::ofstream outfile(kFilename, std::ios_base::app);
    std::ofstream potential_file(kPotentialFilename, std::ios_base::app);
    std::ofstream momentum_file(kMomentumFilename, std::ios_base::app);

    Particle* particles = (Particle*)malloc(sizeof(Particle) * kNumParticles);
    Particle* d_particles;

    std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution_x(-50, 50);
    std::uniform_real_distribution<float> distribution_y(-50, 50);
    std::uniform_real_distribution<float> distribution_z(-2, 2); // top
    std::uniform_real_distribution<float> distribution_mass(1e9, 1e9);
    std::uniform_real_distribution<float> distributaion_vel_x(-0.1, 0.1);
    std::uniform_real_distribution<float> distributaion_vel_y(-0.1, 0.1);

    for (int i = 0; i < kNumParticles; i++) {
        Vec3 rand_pos = {distribution_x(gen), distribution_y(gen), distribution_z(gen)};
        particles[i] = {
            {rand_pos, {distributaion_vel_x(gen), distributaion_vel_y(gen), 0}},
            distribution_mass(gen),
            10
        };
    }

    cudaMalloc((void**)&d_particles, sizeof(Particle) * kNumParticles);
    cudaMemcpy(d_particles, particles, sizeof(Particle) * kNumParticles, cudaMemcpyHostToDevice);

    // get start value
    float start_potential_energy = 0;
    float start_kinetic_energy = 0;
    {
        CalculatePotential<<<kGridDim, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();
        CalculateKinetic<<<kGridDim, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();
        cudaMemcpy(particles, d_particles, sizeof(Particle) * kNumParticles, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int i = 0; i < kNumParticles; i++) {
            start_potential_energy += particles[i].potential_energy / 2;
            start_kinetic_energy += particles[i].kinetic_energy;
        }
    }

    for (int timestep = 0; timestep < kNumTimesteps; timestep++) {
        DeriveKS<1><<<kGridDim, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();
        DeriveKS<2><<<kGridDim, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();
        DeriveKS<3><<<kGridDim, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();
        DeriveKS<4><<<kGridDim, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();
        UpdatePositionVelocity<<<kGridDim, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();

        if (timestep % kNumTimestepsSnapshot == kNumTimestepsSnapshot-1) {
            CalculatePotential<<<kGridDim, kBlockDim>>>(d_particles);
            cudaDeviceSynchronize();
            CalculateKinetic<<<kGridDim, kBlockDim>>>(d_particles);
            cudaDeviceSynchronize();
            cudaMemcpy(particles, d_particles, sizeof(Particle) * kNumParticles, cudaMemcpyDeviceToHost);
            float potential_energy = 0;
            float kinetic_energy = 0;
            Vec3 total_momentum = {0, 0, 0};
            for (int i = 0; i < kNumParticles; i++) {
                potential_energy += particles[i].potential_energy / 2;
                kinetic_energy += particles[i].kinetic_energy;
                total_momentum = total_momentum + particles[i].state.vel * particles[i].mass;
            }

            std::cout << timestep << std::endl;
            potential_file << timestep * kDeltaT << " " << potential_energy-start_potential_energy << " " << kinetic_energy-start_kinetic_energy << " "<< potential_energy-start_potential_energy + kinetic_energy-start_kinetic_energy << std::endl;
            momentum_file << timestep * kDeltaT << " " << total_momentum.x << " " << total_momentum.y << " " << total_momentum.z << " " << GetNorm(total_momentum) << std::endl;
            for (int i = 0; i < kNumParticles; i++) {
                outfile << particles[i].state.pos.x << " " << particles[i].state.pos.y << " " << particles[i].state.pos.z << "\n";
            }
            outfile << "\n";
        }
    }
}
