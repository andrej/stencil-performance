#include <string>

template<typename a, typename b, typename c=std::string>
class Grid {
    public:
    Grid() {}
    virtual void import() = 0;
};

template<typename a, typename c=std::string>
class Coord3Base : public Grid<a, std::string, c> {
    public:
    Coord3Base() : Grid<a, std::string, c>() {}
    void import() final {
        return;
    }
};

template<typename a, typename c=std::string>
class RegularGrid3D : public Coord3Base<a, c> {
    public:
    RegularGrid3D() : Coord3Base<a, c>() {}
};

template<typename a>
class CudaRegularGrid3D : public RegularGrid3D<a, std::string> {
    public:
    CudaRegularGrid3D() : RegularGrid3D<a, std::string>() {};
};

int main() {
    Grid<std::string, std::string> *g = new CudaRegularGrid3D<std::string>();
}