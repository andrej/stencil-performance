#include <malloc.h>

template<typename value_t>
class test {
    public:
    test() {

    }
    value_t *data;
    value_t get() {
        return this->data[0];
    }
    value_t& operator[](const int i) {
        return this->data[i];
    }
};

int main(int argc, char** argv){
    test<double> t();
    test<double> *y=new test<double>();
    y->data = (double *)malloc(sizeof(double)*10);
    y->get();
    double x = (*y)[8];
}