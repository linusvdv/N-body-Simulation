#include <cuda.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <random>


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

    Vec3 operator/(const Vec3& rhs) const {
        return {x / rhs.x, y / rhs.y, z / rhs.z};
    }

    bool operator==(const Vec3& rhs) const {
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


int main() {
    const int block_dim = 64;
    const int num_particles = 128; // easy if multiple from block_dim

    Particle* particles = (Particle*)malloc(sizeof(Particle) * num_particles);
    Particle* c_particles;

    std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution_x(-10, 10);
    std::uniform_real_distribution<float> distribution_y(-10, 10);
    std::uniform_real_distribution<float> distribution_z(-2, 2); // top

    for (int i = 0; i < num_particles; i++) {
        Vec3 rand_pos = {distribution_x(gen), distribution_y(gen), distribution_z(gen)};
        particles[i] = {rand_pos, {0, 0, 0}, {0, 0, 0}};
    }

}
