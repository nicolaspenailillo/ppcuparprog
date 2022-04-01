#include "MPIHandler.hpp"
#include <mpi.h>

const char* MPIHandler::NotInited::what() const noexcept {
    return "ProgramArgs singleton class was not inited with the program args";
}

MPIHandler::~MPIHandler() {
    if(MPI::Is_initialized() && !MPI::Is_finalized()) {
        MPI::Finalize();
    }
}

MPIHandler* MPIHandler::getInstance() {
    static MPIHandler instance;
    return &instance;
}

void MPIHandler::setArgs(int argc, char** argv) {
    args = std::make_pair(argc, argv);
}

void MPIHandler::handleMPIResource() const {
    if(args) {
        auto[argc, argv] = args.value();
        if(!MPI::Is_initialized()) {
            MPI::Init(argc, argv);
        }
    }
    throw NotInited{};
}