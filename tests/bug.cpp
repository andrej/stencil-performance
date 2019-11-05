struct Foo { 
    int a;
};

template<typename T>
class Bar {
    public:
    Foo baz;
    Foo fun() {
        return this->baz = { .a = 42 };
    }
};

int main() {
    Bar<int> *b = new Bar<int>();
    b->fun();
}