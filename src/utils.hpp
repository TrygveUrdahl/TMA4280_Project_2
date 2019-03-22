#pragma once

// Initialize vector with all values initialized to 0
// Input
// n: size of vector
std::vector<double> makeVector(int n);

// TODO: make memory contiguous?
// Input
// n1: first dimension of matrix
// n2: second dimension of matrix
std::vector< std::vector<double> > makeMatrix(int n1, int n2);

// Calculate transpose of matrix in-place
// Input
// mat: matrix to transpose in-place
void transpose(std::vector< std::vector<double> > &mat);

// Calculate right hand side from passed function pointer
// Input
// function: pointer to function for evaluating rhs
// x: first parameter of function
// y: second paramenter of function
double rhs(double (*function)(double, double), double x, double y);

// Export matrix to file
// Input
// mat: matrix to export
void exportMatrix(std::vector< std::vector<double> > &mat);
