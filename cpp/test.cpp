#include "classifier.h"
#include <iostream>

int main() {
  std::string expected = "0.16";
  char result[32];
  GNB classifier;
  std::sprintf(result, "%.2f", classifier.gaussian(1,1,1));
  if (expected.compare(result) != 0) {
    std::cout << "Expected 0.16, got " << result << "\n";
    return -1;
  }

  std::cout << "Done\n";
  return -0;
}