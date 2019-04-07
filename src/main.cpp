// #define testtranspose
// #define printdebug

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "mpi.h"
#include "omp.h"

// Own include files
#include "extern.hpp"
#include "utils.hpp"
#include "structs.hpp"


int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  // Setup MPI variables
  int rank, size;
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  MPI_Comm_size(myComm, &size);
  MPI_Comm_rank(myComm, &rank);

  if (argc < 2) {
    if (rank == 0) {
      std::cout << "Requires argument(s): " << std::endl;
      std::cout << "\tint n: size of grid (must be power of two)" << std::endl;
      std::cout << "\tint r: which right hand side to use (0-4, defaults to 0)" << std::endl;
      std::cout << "\tint t: upper limit of number of OpenMP threads to use (defaults to 1)" << std::endl;
      }
    MPI_Abort(myComm, MPI_ERR_ARG);
    return 1;

  }
  int n = atoi(argv[1]);
  int p = 0;
  if (argc > 2) p = atoi(argv[2]);
  int t = 1;
  if (argc > 3) t = atoi(argv[3]);
  omp_set_num_threads(t);

#ifndef testtranspose
  if (!((n != 0) && ((n &(n - 1)) == 0))) {
    std::cout << "\"n\" must be power of two! " << std::endl;
    MPI_Abort(myComm, MPI_ERR_DIMS);
    return 1;
  }

  if (rank == 0) {
    if (p == 0) {
			std::cout << "Rhs: f(x, y) = 1" << std::endl;
		}
		else if (p == 1) {
			std::cout << "Rhs: 4 single points of value 1 or -1, otherwise 0" << std::endl;
		}
		else if (p == 2) {
			std::cout << "Rhs: f(x, y) = 1/(x^2 + y^2)" << std::endl;
		}
		else if (p == 3) {
			std::cout << "Rhs: f(x, y) = exp(x * y)" << std::endl;
		}
		else if (p == 4) {
			std::cout << "Rhs: f(x, y) = sin(2PI*x) * sin(2PI*y)" << std::endl;
		}
		else {
			std::cout << "DEFAULTS Rhs: f(x, y) = 1" << std::endl;
		}
  }

#endif // testtranspose
#ifdef printdebug
  if (rank == 0) std::cout << "Starting initializations. " << std::endl;
#endif // printdebug
                                          // Grid points per direction is n + 1
  int m = n - 1;                          // Degrees of freedom in each direction
  double h = 1.0/n;                       // Mesh size / step size
  std::vector<int> nPerRankVec(size, 0);  // Cols per rank
  int nn = 4 * n;

  // Initialize nPerRankVec
  for (int i = 0; i < size; i++) {
    nPerRankVec.at(i) = m/size;
  }
  // Distribute the rest of columns
  int rest = m % size;
  for (int i = 0; i < rest; i++) {
    nPerRankVec.at(size - i - 1)++;
  }

  int nPerRank = nPerRankVec.at(rank); // For simplicity typing

  // Setup matrixes and send/recv buffers
  matrix_t b = makeMatrix(nPerRank, m);
  matrix_t bt = makeMatrix(nPerRank, m);
  vector_t xAxis = makeVector(nPerRank);
  vector_t yAxis = makeVector(n + 1);
  vector_t z = makeVector(nn * omp_get_max_threads());
  vector_t diag = makeVector(n);
  vector_t send = makeVector(nPerRank * m);
  vector_t recv = makeVector(nPerRank * m);
  std::vector<int> bsize(size, 0);
  std::vector<int> bsizegather(size, 0);
  std::vector<int> displacement(size, 0);
  std::vector<int> displacementgather(size, 0);

  // Initialize vectors for transpose logic and final gather
  for (int i = 0; i < size; i++) {
    bsize.at(i) = nPerRankVec.at(rank) * nPerRankVec.at(i);
    bsizegather.at(i) = nPerRankVec.at(i) * m;
  }

  for (int i = 1; i < size; i++) {
    displacement.at(i) = displacement.at(i - 1) + bsize.at(i - 1);
    displacementgather.at(i) = displacementgather.at(i - 1) + bsizegather.at(i - 1);
  }

#ifdef testtranspose
  testTranspose(bt, b, m, bsize, bsizegather, displacement, displacementgather, nPerRankVec, rank, size, myComm, send, recv);
#endif // testtranspose

#ifndef testtranspose
  // Create local x and y axis index vectors for convenience
  for (int i = 0; i < nPerRank; i++) {
    xAxis.vec.at(i) = (i + 1 + nPerRank*rank) * h;
  }
  for (int i = 0; i < n + 1; i++) {
    yAxis.vec.at(i) = (i + 1) * h;
  }

  for (int i = 0; i < n; i++) {
    diag.vec.at(i) = 2.0 * (1.0 - cos((i + 1) * M_PI / n));
  }

  for (int i = 0; i < nPerRank; i++) {
    for (int j = 0; j < n; j++) {
      double* elem = b.vec.data() + matIdx(b, i, j);
      *elem = h * h * rhs(rhsPicker(p, rank), xAxis.vec.at(i), yAxis.vec.at(j));
    }
  }
#ifdef printdebug
  if (rank == 0) std::cout << "All initializations complete. " << std::endl;
  if (rank == 0) std::cout << "Starting timer... " << std::endl;
#endif // printdebug
  auto start = std::chrono::high_resolution_clock::now();
#ifdef printdebug
  if (rank == 0) std::cout << "First fst starting... " << std::endl;
#endif // printdebug
////////////////////////////////////////////////////////////////////////////////
  // Start solving, one column FST per iteration
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fst_(b.vec.data() + (m * i), &n, z.vec.data() + omp_get_thread_num() * nn, &nn);
  }
#ifdef printdebug
  if (rank == 0) std::cout << "First fst done... " << std::endl;
  if (rank == 0) std::cout << "First transpose starting... " << std::endl;
#endif // printdebug

  // Transpose
  transpose(bt, b, send, recv, nPerRankVec, bsize, displacement, m, rank, size, myComm);
#ifdef printdebug
  if (rank == 0) std::cout << "First transpose done... " << std::endl;
  if (rank == 0) std::cout << "First fstinv starting... " << std::endl;
#endif // printdebug

  // Inverse FST per column
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fstinv_(bt.vec.data() + (m * i), &n, z.vec.data() + omp_get_thread_num() * nn, &nn);
  }
#ifdef printdebug
  if (rank == 0) std::cout << "First fstinv done... " << std::endl;
  if (rank == 0) std::cout << "Middle step starting... " << std::endl;
#endif // printdebug

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    for (int j = 0; j < m; j++) {
      double* elem = bt.vec.data() + matIdx(b, i, j);
      *elem /= diag.vec.at(i + rank * nPerRank) + diag.vec.at(j);
    }
  }

#ifdef printdebug
  if (rank == 0) std::cout << "Middle step done... " << std::endl;
  if (rank == 0) std::cout << "Second fst starting... " << std::endl;
#endif // printdebug

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fst_(bt.vec.data() + (m * i), &n, z.vec.data() + omp_get_thread_num() * nn, &nn);
  }
#ifdef printdebug
  if (rank == 0) std::cout << "Second fst done... " << std::endl;
  if (rank == 0) std::cout << "Second transpose starting... " << std::endl;
#endif // printdebug
  // Transpose
  transpose(b, bt, send, recv, nPerRankVec, bsize, displacement, m, rank, size, myComm);
#ifdef print
  if (rank == 0) std::cout << "Second transpose done... " << std::endl;
  if (rank == 0) std::cout << "Second fstinv starting... " << std::endl;
#endif // printdebug
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fstinv_(b.vec.data() + (m * i), &n, z.vec.data() + omp_get_thread_num() * nn, &nn);
  }
  auto end = std::chrono::high_resolution_clock::now();
#ifdef printdebug
  if (rank == 0) std::cout << "Second fstinv done... " << std::endl;
  if (rank == 0) std::cout << "Gather starting... " << std::endl;
#endif // printdebug
  matrix_t result = makeMatrix(m,m);
  gatherMatrix(result, b, bsizegather, displacementgather, rank, myComm);
#ifdef printdebug
  if (rank == 0) std::cout << "Gather done... " << std::endl;
#endif // printdebug
  if(rank == 0) exportMatrix(result);

  std::chrono::duration<double> diff = end - start;
  if(rank == 0) std::cout << "Time taken for solving: " << diff.count() << "s" << std::endl;
#endif // testtranspose

  double u_max = 0.0;
  for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < m; j++) {
          u_max = u_max > fabs(result.vec.at(matIdx(b, i, j))) ? u_max : fabs(result.vec.at(matIdx(b, i, j)));
      }
  }
  if (rank == 0) std::cout << "u_max = " << u_max << std::endl;

  MPI_Finalize();
  return 0;
}
