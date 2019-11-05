#include <stdio.h>

class Superclass {
    public:
    virtual void foo() = 0;
};

class Subclass : public Superclass{
    void foo() {
        printf("Hi");
    }
};

int main() {
    Superclass *x = new Subclass();
    x->foo();
}