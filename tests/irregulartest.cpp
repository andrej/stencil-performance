#include <stdio.h>
#include <assert.h>
#include "grids/unstructured.cu"
#include "grids/regular.cu"

int main(int argc, char**argv) {
    // Set up two identical regular grids, one as unstructured and one as regular (reference)
    coord3 size(2,2,2);
    UnstructuredGrid3D<double> *grd = UnstructuredGrid3D<double>::create_regular(size);
    RegularGrid3D<double> *strucgrd = new RegularGrid3D<double>(size);
    for(int x=0; x<grd->dimensions.x; x++) {
        for(int y=0; y<grd->dimensions.y; y++) {
            for(int z=0; z<grd->dimensions.z; z++) {
                grd->set(coord3(x, y, z), (double)x+y*10+z*100);
                strucgrd->set(coord3(x, y, z), (double)x+y*10+z*100);
            }
        }
    }
    // Assert that values at identical coordinates are equal
    for(int x=0; x<grd->dimensions.x; x++) {
        for(int y=0; y<grd->dimensions.y; y++) {
            for(int z=0; z<grd->dimensions.z; z++) {
                printf("(%2d, %2d, %2d: %3f (@grd%3d), %3f (@strucgrd%3d))  \n", 
                       x, y, z, 
                       (*grd)[coord3(x, y, z)], 
                       grd->index(coord3(x, y, z)),
                       (*strucgrd)[coord3(x, y, z)],
                       strucgrd->index(coord3(x, y, z)));
                assert((*grd)[coord3(x, y, z)] == (*strucgrd)[coord3(x, y, z)]);
            }
        }
    }
    // Assert that values at identical neighbors are equal
    coord3 cur = coord3(0, 0, 0);
    for(int i=0; i<grd->dimensions.x*grd->dimensions.y*grd->dimensions.z-1; i++) {
        coord3 neigh(1, 0, 0);
        if(i % grd->dimensions.x == grd->dimensions.x-1) {
            if((i/grd->dimensions.x) % (grd->dimensions.y) == grd->dimensions.z-1) {
                cur.x = 0;
                cur.y = 0;
                neigh = coord3(0, 0, 1);
            } else {
                cur.x = 0;
                neigh = coord3(0, 1, 0);
            }
        }
        coord3 nextcur = cur+neigh;
        printf("%2d%+2d, %2d%+2d, %2d%+2d: %3f (@grd%3d), %3f (@strucgrd%3d)\n", 
               cur.x, neigh.x, cur.y, neigh.y, cur.z, neigh.z, 
               grd->data[grd->neighbor(cur, neigh)], grd->neighbor(cur, neigh),
               strucgrd->data[strucgrd->neighbor(cur, neigh)], strucgrd->neighbor(cur, neigh));
        //assert(grd->data[grd->neighbor(nextcur, neigh)] == strucgrd->data[strucgrd->neighbor(nextcur, neigh)]);
        cur = nextcur;
    }
    return 0;
}