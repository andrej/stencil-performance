#include "stdio.h"
int main() {
    int x = do{ 7; }while(0);
    printf("%d\n", x);
}