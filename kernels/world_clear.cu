#include "qcuda.h"
// WeightWorld: framebuffer words 16..79; state words 0..15 remain untouched.
extern "C" __global__ void world_clear(int *world) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  world[16 + i] = 0; // launch exactly 64 threads
}
