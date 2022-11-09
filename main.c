#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "rngs.h"
#include "utils.h"

// A = P_size x Q_size   -- matrix
// B = Q_size x R_size

// C = R_size x S_size
// D = S_size x T_size

// G = P_size x T_size matrix ? where
// G = (A.B) . (C.D) =>   (A.B) = E and (C.D) = F;
// G = E.F

void kernel_3mm(int P_size, int R_size, int Q_size, int T_size, int S_size,
                double E[P_size][R_size], double A[P_size][Q_size],
                double B[Q_size][R_size],

                double F[R_size][T_size], double C[R_size][S_size],
                double D[S_size][T_size],

                double G[P_size][T_size]) {

#pragma scop
  // matrix multiply
  // E = A * B
  for (int P_indx = 0; P_indx < P_size; P_indx++) {
    for (int R_indx = 0; R_indx < R_size; R_indx++) {
      E[P_indx][R_indx] = 0.0;
      for (int Q_indx = 0; Q_indx < Q_size; ++Q_indx)
        E[P_indx][R_indx] += A[P_indx][Q_indx] * B[Q_indx][R_indx];
    }
  }

  // F = C * D
  for (int R_indx = 0; R_indx < R_size; R_indx++)
    for (int T_indx = 0; T_indx < T_size; T_indx++) {
      F[R_indx][T_indx] = 0.0;
      for (int S_indx = 0; S_indx < S_size; ++S_indx)
        F[R_indx][T_indx] += C[R_indx][S_indx] * D[S_indx][T_indx];
    }
  //  G = E * F
  for (int P_indx = 0; P_indx < P_size; P_indx++)
    for (int T_indx = 0; T_indx < T_size; T_indx++) {
      G[P_indx][T_indx] = 0.0;
      for (int R_indx = 0; R_indx < R_size; ++R_indx)
        G[P_indx][T_indx] += E[P_indx][R_indx] * F[R_indx][T_indx];
    }
#pragma endscop
}

void kernel_3mm_paralel(int P_size, int R_size, int Q_size, int T_size,
                        int S_size, double E[P_size][R_size],
                        double A[P_size][Q_size], double B[Q_size][R_size],

                        double F[R_size][T_size], double C[R_size][S_size],
                        double D[S_size][T_size],

                        double G[P_size][T_size]) {

  // matrix multiply
  // E = A * B
#pragma scop

#pragma omp parallel for
  for (int P_indx = 0; P_indx < P_size; P_indx++) {
    for (int R_indx = 0; R_indx < R_size; R_indx++) {
      E[P_indx][R_indx] = 0.0;
      for (int Q_indx = 0; Q_indx < Q_size; ++Q_indx)
        E[P_indx][R_indx] += A[P_indx][Q_indx] * B[Q_indx][R_indx];
    }
  }

  // F = C * D
#pragma omp parallel for
  for (int R_indx = 0; R_indx < R_size; R_indx++)
    for (int T_indx = 0; T_indx < T_size; T_indx++) {
      F[R_indx][T_indx] = 0.0;
      for (int S_indx = 0; S_indx < S_size; ++S_indx)
        F[R_indx][T_indx] += C[R_indx][S_indx] * D[S_indx][T_indx];
    }
//  G = E * F
#pragma omp parallel for
  for (int P_indx = 0; P_indx < P_size; P_indx++)
    for (int T_indx = 0; T_indx < T_size; T_indx++) {
      G[P_indx][T_indx] = 0.0;
      for (int R_indx = 0; R_indx < R_size; ++R_indx)
        G[P_indx][T_indx] += E[P_indx][R_indx] * F[R_indx][T_indx];
    }
#pragma endscop

}

void kernel_3mm_paralel_2tread(int P_size, int R_size, int Q_size, int T_size,
                        int S_size, double E[P_size][R_size],
                        double A[P_size][Q_size], double B[Q_size][R_size],

                        double F[R_size][T_size], double C[R_size][S_size],
                        double D[S_size][T_size],

                        double G[P_size][T_size]) {

  // matrix multiply
  // E = A * B
#pragma omp parallel for num_threads(2)
  for (int P_indx = 0; P_indx < P_size; P_indx++) {
    for (int R_indx = 0; R_indx < R_size; R_indx++) {
      E[P_indx][R_indx] = 0.0;
      for (int Q_indx = 0; Q_indx < Q_size; ++Q_indx)
        E[P_indx][R_indx] += A[P_indx][Q_indx] * B[Q_indx][R_indx];
    }
  }

  // F = C * D
#pragma omp parallel for num_threads(2)
  for (int R_indx = 0; R_indx < R_size; R_indx++)
    for (int T_indx = 0; T_indx < T_size; T_indx++) {
      F[R_indx][T_indx] = 0.0;
      for (int S_indx = 0; S_indx < S_size; ++S_indx)
        F[R_indx][T_indx] += C[R_indx][S_indx] * D[S_indx][T_indx];
    }
//  G = E * F
#pragma omp parallel for num_threads(2)
  for (int P_indx = 0; P_indx < P_size; P_indx++)
    for (int T_indx = 0; T_indx < T_size; T_indx++) {
      G[P_indx][T_indx] = 0.0;
      for (int R_indx = 0; R_indx < R_size; ++R_indx)
        G[P_indx][T_indx] += E[P_indx][R_indx] * F[R_indx][T_indx];
    }
}

void kernel_3mm_paralel_4tread(int P_size, int R_size, int Q_size, int T_size,
                        int S_size, double E[P_size][R_size],
                        double A[P_size][Q_size], double B[Q_size][R_size],

                        double F[R_size][T_size], double C[R_size][S_size],
                        double D[S_size][T_size],

                        double G[P_size][T_size]) {

  // matrix multiply
  // E = A * B
#pragma omp parallel for num_threads(4)
  for (int P_indx = 0; P_indx < P_size; P_indx++) {
    for (int R_indx = 0; R_indx < R_size; R_indx++) {
      E[P_indx][R_indx] = 0.0;
      for (int Q_indx = 0; Q_indx < Q_size; ++Q_indx)
        E[P_indx][R_indx] += A[P_indx][Q_indx] * B[Q_indx][R_indx];
    }
  }

  // F = C * D
#pragma omp parallel for num_threads(4)
  for (int R_indx = 0; R_indx < R_size; R_indx++)
    for (int T_indx = 0; T_indx < T_size; T_indx++) {
      F[R_indx][T_indx] = 0.0;
      for (int S_indx = 0; S_indx < S_size; ++S_indx)
        F[R_indx][T_indx] += C[R_indx][S_indx] * D[S_indx][T_indx];
    }
//  G = E * F
#pragma omp parallel for num_threads(4)
  for (int P_indx = 0; P_indx < P_size; P_indx++)
    for (int T_indx = 0; T_indx < T_size; T_indx++) {
      G[P_indx][T_indx] = 0.0;
      for (int R_indx = 0; R_indx < R_size; ++R_indx)
        G[P_indx][T_indx] += E[P_indx][R_indx] * F[R_indx][T_indx];
    }
}


int main() {

  PlantSeeds(time(0));

  const int P_size = 900;
  const int Q_size = 500;
  const int R_size = 700;
  const int T_size = 400;
  const int S_size = 600;

  double(*A)[P_size][Q_size];
  A = (double(*)[P_size][Q_size])malloc(P_size * Q_size * sizeof(double));

  double(*B)[Q_size][R_size];
  B = (double(*)[Q_size][R_size])malloc(Q_size * R_size * sizeof(double));

  double(*E)[P_size][R_size];
  E = (double(*)[P_size][R_size])malloc(P_size * R_size * sizeof(double));

  double(*C)[R_size][S_size];
  C = (double(*)[R_size][S_size])malloc(R_size * S_size * sizeof(double));

  double(*D)[S_size][T_size];
  D = (double(*)[S_size][T_size])malloc(S_size * T_size * sizeof(double));

  double(*F)[R_size][T_size];
  F = (double(*)[R_size][T_size])malloc(R_size * T_size * sizeof(double));

  double(*G)[P_size][T_size];
  G = (double(*)[P_size][T_size])malloc(P_size * T_size * sizeof(double));

  fill_mtrx_rand_val(P_size, Q_size, A);
  fill_mtrx_rand_val(Q_size, R_size, B);
  fill_mtrx_rand_val(R_size, S_size, C);
  fill_mtrx_rand_val(S_size, T_size, D);

  //
  struct timeval start_time;
  struct timeval finish_time;

  struct timeval start_time_omp;
  struct timeval finish_time_omp;

  gettimeofday(&start_time, 0);

  kernel_3mm(P_size, R_size, Q_size, T_size, S_size, E, A, B, F, C, D, G);

  gettimeofday(&finish_time, 0);

  float elapsedTime = elapsed_msecs(start_time, finish_time);
  printf("run without omp wersion \n");
  printf("Elapsed Time: %f milliseconds\n", elapsedTime);

  gettimeofday(&start_time_omp, 0);

  kernel_3mm_paralel(P_size, R_size, Q_size, T_size, S_size, E, A, B, F, C, D,
                     G);

  gettimeofday(&finish_time_omp, 0);
  float elapsedTime_omp = elapsed_msecs(start_time_omp, finish_time_omp);
  printf("run with omp wersion \n");
  printf("Elapsed Time: %f milliseconds\n", elapsedTime_omp);


  struct timeval start_time_omp_2thread;
  struct timeval finish_time_omp_2thread;

  gettimeofday(&start_time_omp_2thread, 0);

  kernel_3mm_paralel_2tread(P_size, R_size, Q_size, T_size, S_size, E, A, B, F, C, D,
                     G);

  gettimeofday(&finish_time_omp_2thread, 0);
  float elapsedTime_omp_2 = elapsed_msecs(start_time_omp_2thread, finish_time_omp_2thread);
  printf("run with omp 2 threads wersion \n");
  printf("Elapsed Time: %f milliseconds\n", elapsedTime_omp_2);

  struct timeval start_time_omp_4thread;
  struct timeval finish_time_omp_4thread;

  gettimeofday(&start_time_omp_4thread, 0);

  kernel_3mm_paralel_4tread(P_size, R_size, Q_size, T_size, S_size, E, A, B, F, C, D,
                     G);

  gettimeofday(&finish_time_omp_4thread, 0);
  float elapsedTime_omp_4 = elapsed_msecs(start_time_omp_4thread, finish_time_omp_4thread);
  printf("run with omp 4 threads wersion \n");
  printf("Elapsed Time: %f milliseconds\n", elapsedTime_omp_4);


  // print_matrix(P_size, T_size, G);

  free(A);
  free(B);
  free(E);
  free(C);
  free(D);
  free(F);
  free(G);
}
