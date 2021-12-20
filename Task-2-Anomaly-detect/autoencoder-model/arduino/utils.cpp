#include <stdlib.h>
#include <math.h>

// Calculate the mean squared error between two arrays
float calc_mae(const float *x, const float *x_hat, const int len) {
  
  float mae = 0;

  // Square difference between each set of elements
  for (int i = 0; i < len; i++) {
    mae += fabs(x[i] - x_hat[i]);
  }

  return mae / len;
}
