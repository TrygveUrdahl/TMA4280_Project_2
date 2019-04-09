#pragma once
// Collection of all rhs functions

double fRhs(double x, double y) {
  return 1;
}

double singularRhs(double x, double y) {
  double val = 1;
  if (x == 0.25 && y == 0.25) return val;
  if (x == 0.25 && y == 0.75) return -val;
  if (x == 0.75 && y == 0.25) return -val;
  if (x == 0.75 && y == 0.75) return val;
  return 0;
}

double invRadialRhs(double x, double y) {
  return 1/(x*x + y*y);
}

double expRhs(double x, double y) {
  return exp(x * y);
}

double trigRhs(double x, double y) {
  return sin(x * 2 * M_PI) * sin(y * 2 * M_PI);
}

double discreteRhs(double x, double y) {
  if (x > 0.25 && y > 0.25) return 1;
  if (x < 0.25 && y < 0.25) return -3;
  return 0;
}
