#pragma once

#include "structs.hpp"


inline int matIdx(matrix_t &mat, int x, int y);

// Initialize vector with all values initialized to 0
// Input
// n: size of vector
vector_t makeVector(int n);

// TODO: make memory contiguous?
// Input
// n1: first dimension of matrix
// n2: second dimension of matrix
matrix_t makeMatrix(int n1, int n2);

// Calculate transpose of matrix in-place
// Input
// mat: matrix to transpose in-place
void transposeSeq(matrix_t &mat);

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

// Generate 1D Laplace operator matrix
// Input
// n: dimension of matrix
matrix_t make1DLaplaceOperator(int n);
