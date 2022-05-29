#include <stdio.h>

int Sub(int a, int b) {
  return a - b;
}

int main(void) {
  int a = 1, b = 2;
  printf("Hello Flex!\n");
  printf("  %d + %d = %d\n", a, b, Add(a, b));
  printf("  %d - %d = %d\n", a, b, Sub(a, b));
  return 0;
}
