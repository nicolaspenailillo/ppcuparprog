#include "gtest/gtest.h"
#include "MPIHandler.hpp"

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    MPIHandler::getInstance()->setArgs(argc, argv);
    return RUN_ALL_TESTS();
}