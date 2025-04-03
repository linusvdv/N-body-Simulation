#include <cuda.h>
#include <driver_types.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <random>


const std::string kFilename = "out.xyz";
const int kBlockDim = 1024;
const int kNumParticles = 10;
__device__ const float kDeltaT = 0.1;
const float kGravitationalConstant = 6.67430e-11;
const int kNumTimesteps = 100000;
const int kNumTimestepsSnapshot = 100;


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

    __device__ Vec3 operator+(const Vec3& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }

    __device__ Vec3 operator-(const Vec3& rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }

    __device__ Vec3 operator*(const Vec3& rhs) const {
        return {x * rhs.x, y * rhs.y, z * rhs.z};
    }

    __device__ Vec3 operator*(const float& rhs) const {
        return {x * rhs, y * rhs, z * rhs};
    }

    __device__ Vec3 operator/(const Vec3& rhs) const {
        return {x / rhs.x, y / rhs.y, z / rhs.z};
    }

    __device__ Vec3 operator/(const float& rhs) const {
        return {x / rhs, y / rhs, z / rhs};
    }

    __device__ bool operator==(const Vec3& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};


struct Particle {
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    float mass;
    float radius;
    float potential_energy = 0;
    float kinetic_energy = 0;
};


__global__ void UpdatePosition(Particle* particles) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= kNumParticles) {
        return;
    }
    particles[index].pos = particles[index].pos + particles[index].vel * kDeltaT;
}


__global__ void UpdateVelocityHalf(Particle* particles) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= kNumParticles) {
        return;
    }
    particles[index].vel = particles[index].vel + particles[index].acc * kDeltaT / 2;
}


__device__ float GetNorm(const Vec3& vec) {
    return std::sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}


__device__ Vec3 GetUnitVector(const Vec3& vec) {
    return vec / GetNorm(vec);
}


__device__ Vec3 GravitationalForce(const Particle& first, const Particle& second) {
    Vec3 direction_unit_vector = GetUnitVector(second.pos - first.pos);
    float force = kGravitationalConstant * first.mass * second.mass / ((GetNorm(second.pos - first.pos) + 0.1) * (GetNorm(second.pos - first.pos) + 0.1));
    return direction_unit_vector * force;
}


__global__ void DeriveAcc(Particle* particles) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= kNumParticles) {
        return;
    }
    particles[index].acc = {0, 0, 0};
    for (int i = 0; i < kNumParticles; i++) {
        if (i == index) {
            continue;
        }
        particles[index].acc = particles[index].acc + GravitationalForce(particles[index], particles[i]) / particles[index].mass;
    }
}


__device__ float GetPotential(const Particle& first, const Particle& second) {
    return kGravitationalConstant * first.mass * second.mass / (GetNorm(second.pos - first.pos) + 0.1);
}


__global__ void CalculatePotential(Particle* particles) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
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
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= kNumParticles) {
        return;
    }
    particles[index].kinetic_energy = 1. / 2 * particles[index].mass * GetNorm(particles[index].vel) * GetNorm(particles[index].vel);
}


int main() {
    {
        std::ofstream outfile;
        outfile.open(kFilename);
        outfile << kNumParticles << " " << kNumTimesteps << " " << kNumTimestepsSnapshot << "\n";
        outfile.close();
    }
    std::ofstream outfile(kFilename, std::ios_base::app);

    Particle* particles = (Particle*)malloc(sizeof(Particle) * kNumParticles);
    Particle* d_particles;

    std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution_x(-100, 100);
    std::uniform_real_distribution<float> distribution_y(-100, 100);
    std::uniform_real_distribution<float> distribution_z(-2, 2); // top
    std::uniform_real_distribution<float> distribution_mass(1e9, 1e9);
    std::uniform_real_distribution<float> distributaion_vel_x(-0, 0);
    std::uniform_real_distribution<float> distributaion_vel_y(-0, 0);

    for (int i = 0; i < kNumParticles; i++) {
        Vec3 rand_pos = {distribution_x(gen), distribution_y(gen), distribution_z(gen)};
        particles[i] = {rand_pos, {distributaion_vel_x(gen), distributaion_vel_y(gen), 0}, {0, 0, 0}, distribution_mass(gen), 10};
    }

    cudaMalloc((void**)&d_particles, sizeof(Particle) * kNumParticles);
    cudaMemcpy(d_particles, particles, sizeof(Particle) * kNumParticles, cudaMemcpyHostToDevice);

    for (int timestep = 0; timestep < kNumTimesteps; timestep++) {
        UpdateVelocityHalf<<<(kNumParticles/kBlockDim)+1, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();
        UpdatePosition<<<(kNumParticles/kBlockDim)+1, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();
        DeriveAcc<<<(kNumParticles/kBlockDim)+1, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();
        UpdateVelocityHalf<<<(kNumParticles/kBlockDim)+1, kBlockDim>>>(d_particles);
        cudaDeviceSynchronize();

        if (timestep % kNumTimestepsSnapshot == kNumTimestepsSnapshot-1) {
            CalculatePotential<<<(kNumParticles/kBlockDim)+1, kBlockDim>>>(d_particles);
            cudaDeviceSynchronize();
            CalculateKinetic<<<(kNumParticles/kBlockDim)+1, kBlockDim>>>(d_particles);
            cudaDeviceSynchronize();
            cudaMemcpy(particles, d_particles, sizeof(Particle) * kNumParticles, cudaMemcpyDeviceToHost);
            float potential_energy = 0;
            float kinetic_energy = 0;
            for (int i = 0; i < kNumParticles; i++) {
                potential_energy += particles[i].potential_energy / 2;
                kinetic_energy += particles[i].kinetic_energy;
            }

            std::cout << timestep << " " << potential_energy << " " << kinetic_energy << " "<< potential_energy + kinetic_energy << std::endl;
            /*for (int i = 0; i < kNumParticles; i++) {
                outfile << particles[i].pos.x << " " << particles[i].pos.y << " " << particles[i].pos.z << "\n";
            }*/
            outfile << "\n";
        }
    }
}
