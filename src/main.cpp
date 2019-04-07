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

double fRhs(double x, double y) {
  return 1;
}

double singularRhs(double x, double y) {
  double val = 10;
  if (x == 0.25 && y == 0.25) return val;
  if (x == 0.25 && y == 0.75) return -val;
  if (x == 0.75 && y == 0.25) return -val;
  if (x == 0.75 && y == 0.75) return val;
  return 0;
}

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
      std::cout << "\tint n: size of grid (power of two)" << std::endl;

      std::cout << "\tint t: number of OpenMP threads to use (defaults to 1)" << std::endl;
      }
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
    return 1;

  }
  int n = atoi(argv[1]);
  int t = 1;
  if (argc > 2) t = atoi(argv[2]);
  // omp_set_num_threads(t);

#ifndef testtranspose
  if (!((n != 0) && ((n &(n - 1)) == 0))) {
    std::cout << "\"n\" must be power of two! " << std::endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_DIMS);
    return 1;
  }
#endif // testtranspose


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
  vector_t z = makeVector(nn);
  vector_t diag = makeVector(n);
  vector_t send = makeVector(nPerRank * m);
  vector_t recv = makeVector(nPerRank * m);
  std::vector<int> bsize(size, 0);
  std::vector<int> bsizegather(size, 0);
  std::vector<int> displacement(size, 0);
  std::vector<int> displacementgather(size, 0);
  std::vector<int> localdisplacement(size, 0);

  // Initialize vectors for transpose logic
  if (rank==0) std::cout << "bsize: ";
  for (int i = 0; i < size; i++) {
    bsize.at(i) = nPerRankVec.at(rank) * nPerRankVec.at(i);
    bsizegather.at(i) = nPerRankVec.at(i) * m;
    if (rank == 0) std::cout << bsize.at(i) << " ";
  }
  if (rank==0) std::cout << std::endl;
  for (int i = 1; i < size; i++) {
    displacement.at(i) = displacement.at(i - 1) + bsize.at(i - 1);
    displacementgather.at(i) = displacementgather.at(i - 1) + bsizegather.at(i - 1);
    localdisplacement.at(i) = localdisplacement.at(i - 1) + nPerRankVec.at(i - 1);
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
      *elem = h * h * rhs(fRhs, xAxis.vec.at(i), yAxis.vec.at(j));
    }
  }
  auto start = std::chrono::high_resolution_clock::now();
#ifdef printdebug
  if (rank == 0) std::cout << "First fst starting... " << std::endl;
#endif // printdebug
  // Start solving, one column FST per iteration
  //#pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fst_(b.vec.data() + (m * i), &n, z.vec.data(), &nn);
  }
#ifdef printdebug
  if (rank == 0) std::cout << "First fst done... " << std::endl;
  if (rank == 0) std::cout << "First transpose starting... " << std::endl;
#endif // printdebug
  // Transpose
  //transpose_p(bt, b, bsize, displacement, nPerRankVec, rank, size, myComm);
  //transpose(bt, b, bsize, displacement, nPerRankVec, rank, size, myComm);
  transpose_3(bt, b, send, recv, nPerRankVec, bsize, displacement, m, rank, size, myComm);
#ifdef printdebug
  if (rank == 0) std::cout << "First transpose done... " << std::endl;
  if (rank == 0) std::cout << "First fstinv starting... " << std::endl;
#endif // printdebug
  // Inverse FST per column
  //#pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fstinv_(bt.vec.data() + (m * i), &n, z.vec.data(), &nn);
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

  //#pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fst_(bt.vec.data() + (m * i), &n, z.vec.data(), &nn);
  }
#ifdef printdebug
  if (rank == 0) std::cout << "Second fst done... " << std::endl;
  if (rank == 0) std::cout << "Second transpose starting... " << std::endl;
#endif // printdebug
  // Transpose
  //transpose_p(b, bt, bsize, displacement, nPerRankVec, rank, size, myComm);
  //transpose(b, bt, bsize, displacement, nPerRankVec, rank, size, myComm);
  transpose_3(b, bt, send, recv, nPerRankVec, bsize, displacement, m, rank, size, myComm);
#ifdef print
  if (rank == 0) std::cout << "Second transpose done... " << std::endl;
  if (rank == 0) std::cout << "Second fstinv starting... " << std::endl;
#endif // printdebug
  //#pragma omp parallel for schedule(static)
  for (int i = 0; i < nPerRank; i++) {
    fstinv_(b.vec.data() + (m * i), &n, z.vec.data(), &nn);
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
#endif // testtranspose

  std::chrono::duration<double> diff = end-start;
  std::cout << "Time taken for solving: " << diff.count() << "s" << std::endl;



  MPI_Finalize();
  return 0;
}
