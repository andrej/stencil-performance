
/** This is a bug in the nvcc compiler.
 * Reported as #2748928 */
    struct Foo { 
        int a;
    };

    template<typename T>
    class Bar {
        public:
        Foo baz;
        void fun() {
            this->baz = { .a = 42 };
        }
    };

    int main() {
        return 7;
        Bar<int> *obj = new Bar<int>();
        //obj->fun();
    }