#include "qcuda.h"
// 4 particles: interleaved x/y at 0..7, signed vx/vy at 8..15.
// Position is a fixed-point integer in [0,127]; each pixel spans 16 units.
extern "C" __global__ void world_advance(int *world) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // exactly 8 threads
  int v = world[8 + i];
  int p = world[i] + v;
  int low = p < 0;
  int high = p > 127;
  world[i] = low ? -p : (high ? 254 - p : p);
  world[8 + i] = (low || high) ? -v : v;
}
