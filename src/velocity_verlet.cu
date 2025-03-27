#include <__clang_cuda_runtime_wrapper.h>
#include <cuda.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>


const int kBlockDim = 64;
const int kNumParticles = 128;
const float kDeltaT = 0.1;
const float kGravitationalConstant = 6.67430e-11;


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

    Vec3 operator+(const Vec3& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }

    Vec3 operator-(const Vec3& rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }

    Vec3 operator*(const Vec3& rhs) const {
        return {x * rhs.x, y * rhs.y, z * rhs.z};
    }

    Vec3 operator*(const float& rhs) const {
        return {x * rhs, y * rhs, z * rhs};
    }

    Vec3 operator/(const Vec3& rhs) const {
        return {x / rhs.x, y / rhs.y, z / rhs.z};
    }

    bool operator==(const Vec3& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
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
    particles[index].vel = particles[index].vel + particles[index].acc * kDeltaT * 0.5;
}


__device__ float GetNorm(const Vec3& vec) {
    return std::sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}


__device__ Vec3 GetUnitVector(const Vec3& vec) {
    return vec / GetNorm(vec);
}


__device__ Vec3 GravitationalForce(const Particle& first, const Particle& second) {
    Vec3 direction_unit_vector = GetUnitVector(second.pos - first.pos);
    float force = kGravitationalConstant * first.mass * second.mass / (GetNorm(second.pos - first.pos) * GetNorm(second.pos - first.pos));
    return direction_unit_vector * force;
}


__global__ void DeriveAcc(Particle* particles) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= kNumParticles) {
        return;
    }
    for (int i = 0; i < kNumParticles; i++) {
        if (i == index) {
            continue;
        }
        particles[index].acc = particles[index].acc + GravitationalForce(particles[index], particles[i]);
    }
}


int main() {
    Particle* particles = (Particle*)malloc(sizeof(Particle) * kNumParticles);
    Particle* c_particles;

    std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution_x(-10, 10);
    std::uniform_real_distribution<float> distribution_y(-10, 10);
    std::uniform_real_distribution<float> distribution_z(-2, 2); // top
    std::uniform_real_distribution<float> distribution_mass(1, 100);

    for (int i = 0; i < kNumParticles; i++) {
        Vec3 rand_pos = {distribution_x(gen), distribution_y(gen), distribution_z(gen)};
        particles[i] = {rand_pos, {0, 0, 0}, {0, 0, 0}, distribution_mass(gen), 10};
    }


}
