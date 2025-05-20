# N-body Simulation using CUDA
[PlanetsOrbiting.webm](https://github.com/user-attachments/assets/4cd53467-1b8f-408d-9ea7-502ea87ac9ca)

N-body simulation parallized with cuda

## run

```bash
cd src
nvcc -std=c++17 -O3 -o velocity_verlet velocity_verlet.cu
./velocity_verlet
python3 render.py
```

