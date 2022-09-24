#ifndef __UTILS_H__
#define __UTILS_H__

#include "rngs.h"
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

float elapsed_msecs(struct timeval s, struct timeval f) {
  return (float)(1000.0 * (f.tv_sec - s.tv_sec) +
                 (0.001 * (f.tv_usec - s.tv_usec)));
}

void print_matrix(int A_size, int B_size, double Matrix[A_size][B_size]) {
  printf("------------------- \n");
  for (int A_indx = 0; A_indx < A_size; ++A_indx) {
    for (int B_indx = 0; B_indx < B_size; ++B_indx) {
      printf("%.3lf  ", Matrix[A_indx][B_indx]);
    }
    printf("\n");
  }
  printf("------------------- \n");
}

void fill_mtrx_rand_val(int A_size, int B_size, double Matrix[A_size][B_size]) {
  for (int A_indx = 0; A_indx < A_size; ++A_indx) {
    for (int B_indx = 0; B_indx < B_size; ++B_indx) {
      Matrix[A_indx][B_indx] = Random();
    }
    printf("\n");
  }
}

#endif // __UTILS_H__