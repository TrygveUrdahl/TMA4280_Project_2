#include <iostream>
#include <vector>
#include <chrono>
#include "mpi.h"
#include "omp.h"


// Own include files
#include "extern.hpp"
#include "utils.hpp"
#include "structs.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  if (argc < 2) {
    std::cout << "Requires argument(s): " << std::endl;
    std::cout << "\tint n: size of grid (power of two)" << std::endl;
    MPI_Finalize();
    return 1;
  }
  int n = atoi(argv[1]);
  if (!((n != 0) && ((n &(n - 1)) == 0))) {
    std::cout << "\"n\" must be power of two! " << std::endl;
    MPI_Finalize();
    return 1;
  }
  // Setup for program parameters completed

                      // grid points per direction is n + 1
  int m = n - 1;      // degrees of freedom in each direction
  double h = 1.0/n;   // mesh size / step size

  matrix_t b = makeMatrix(m, m);
  matrix_t bt = makeMatrix(m, m);




  auto start = std::chrono::high_resolution_clock::now();
  auto mat = make1DLaplaceOperator(n);
  auto end = std::chrono::high_resolution_clock::now();
  printMatrix(mat);
  //transposeSeq(mat);
  //printMatrix(mat);
  std::chrono::duration<double> diff = end-start;
  std::cout << "Time taken to create Laplace Operator: " << diff.count() << "s" << std::endl;





  MPI_Finalize();
  return 0;
}
