#include "pi.hpp"
#include "bigArray.hpp"
#include "mpiCart.hpp"

int main(int argc, char* argv[]) {
    // return pi(argc, argv);
    // return bigArray(argc, argv);
    return mpiCart(argc, argv);
}