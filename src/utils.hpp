#pragma once

#include "structs.hpp"

// Return id of element (x, y) in matrix
// Input
// mat: matrix to look up element in
// x, y: indices
int matIdx(matrix_t &mat, int x, int y);

// Initialize vector with all values initialized to 0
// Input
// n: size of vector
vector_t makeVector(int n);

// Initialize matrix with all values initialized to 0
// Input
// n1: first dimension of matrix
// n2: second dimension of matrix
matrix_t makeMatrix(int n1, int n2);

// Calculate right hand side from passed function pointer
// Input
// function: pointer to function for evaluating rhs
// x: first parameter of function
// y: second paramenter of function
double rhs(double (*function)(double, double), double x, double y);

// Export matrix to file
// Input
// mat: matrix to export
// fname: name of output file (optional)
void exportMatrix(matrix_t &mat, std::string fname = "../output/mat.txt");

// Print matrix to console for debug
// Input
// mat: matrix to print
void printMatrix(matrix_t &mat);

// Parallel transpose
void transpose(matrix_t &bt, matrix_t &b, vector_t &send, vector_t &recv,
                  std::vector<int> &nPerRankVec, std::vector<int> &bsize,
                  std::vector<int> &displacement, int n, int rank, int size,
                  MPI_Comm myComm);

// Gather all b matrices to result on root node 0
// Input
void gatherMatrix(matrix_t &result, matrix_t &b, std::vector<int> &bsize,
                std::vector<int> &displacement, int rank, MPI_Comm myComm);

// Function to test the transpose, printing a dummy matrix before and after transpose
void testTranspose(matrix_t &bt, matrix_t &b, int m, std::vector<int> &bsize,
                  std::vector<int> &bsizegather, std::vector<int> &displacement,
                  std::vector<int> &displacementgather, std::vector<int> &nPerRankVec,
                  int rank, int size, MPI_Comm myComm, vector_t &send, vector_t &recv);

// Fill a matrix with dummy values
void fillMatrix(matrix_t &b, std::vector<int> &displacementgather, int n1,
                  int n2, int rank);

// Function to pick which rhs function to use (yay for trailing return type)
auto rhsPicker(int p) -> double(*)(double, double);
