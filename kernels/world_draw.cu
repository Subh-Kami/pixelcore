#include "qcuda.h"
// Scatter rasterizer. Four threads write color IDs directly into the framebuffer.
// Overlaps resolve in the circuit's deterministic lane order: highest ID wins.
extern "C" __global__ void world_draw(int *world) {
  int i = threadIdx.x; // exactly one 4-lane warp
  int x = world[2 * i] / 16;
  int y = world[2 * i + 1] / 16;
  int pixel = (8 * y + x) & 63;
  world[16 + pixel] = i + 1;
}
