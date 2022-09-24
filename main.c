#include <stdlib.h>
#include <stdio.h>

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
                double E[P_size][R_size],
                double A[P_size][Q_size], 
                double B[Q_size][R_size], 
                double F[R_size][T_size],
                double C[R_size][S_size], 
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

int main() {

   PlantSeeds(time(0));

  const int A_size = 4; 
  const int B_size = 5;   
  double  (*A)[A_size][B_size];
  A = (double (*)[A_size][B_size]) malloc (A_size * B_size * sizeof(double));  

  fill_mtrx_rand_val(A_size, B_size, A);
  print_matrix(A_size, B_size, A);

  free(A);


}
