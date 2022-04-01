#include "MPIHandler.hpp"
#include "fmt/core.h"
#include <mpi.h>

const char* MPIHandler::NotInited::what() const noexcept { return "ProgramArgs singleton class was not inited with the program args"; }

MPIHandler::~MPIHandler() {
    MPI_Initialized(&inited);
    MPI_Finalized(&finalized);
    if (inited && !finalized) {
        MPI_Finalize();
    }
}

MPIHandler* MPIHandler::getInstance() {
    static MPIHandler instance;
    return &instance;
}

void MPIHandler::setArgs(int argc, char** argv) { args = std::make_pair(argc, argv); }

void MPIHandler::handleMPIResource() {
    if (args) {
        auto [argc, argv] = args.value();
        MPI_Initialized(&inited);
        if (!inited) {
            MPI_Init(&argc, &argv);
        }
    } else {
        throw NotInited{};
    }
}