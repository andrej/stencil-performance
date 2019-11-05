#include "stdio.h"
#include "coord3.cu"

void printcoord(coord3 *coord) {
    printf("(%d, %d, %d)\n", coord->x, coord->y, coord->z);
}

int main(int argc, char **argv) {

    coord3 A = coord3(1.0,2.0,3.0);
    coord3 B = coord3(1,1,2);

    printcoord(&A);
    printcoord(&B);

    coord3 C = A-B;
    printcoord(&C);

    coord3 D = 3*A;
    printcoord(&D);

}