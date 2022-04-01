#include "simulator.hpp"
#include "fmt/core.h"
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <mpi.h>
#include "MPIHandler.hpp"

void Simulator::setPrinting(bool toPrint) { printing = toPrint; }

void Simulator::initU() {
    //initialization doesnt need parallalel programming
    countMap["initU"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i <= (grid - 1); i++) {
        u[(i) * (grid + 1) + grid] = 1.0;
        u[(i) * (grid + 1) + grid - 1] = 1.0;
        for (SizeType j = 0; j < (grid - 1); j++) {
            u[(i) * (grid + 1) + j] = 0.0;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["initU"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["initU"] =  (grid-1)*(4) + (grid-1)*(grid-1)*(4);
}

void Simulator::initV() {
    //initialization doesnt need parallalel programming
    countMap["initV"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i <= (grid); i++) {
        for (SizeType j = 0; j <= (grid); j++) {
            v[(i)*(grid + 1) + j] = 0.0;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["initV"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["initV"] =  (grid)*(grid-1)*(4);
}

void Simulator::initP() {
    //initialization doesnt need parallalel programming
    countMap["initP"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i <= (grid); i++) {
        for (SizeType j = 0; j <= (grid); j++) {
            p[(i) * (grid + 1) + j] = 1.0;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["initP"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["initP"] =  (grid)*(grid)*(4);
}

void Simulator::solveUMomentum(const FloatType Re) {
    countMap["solveUMomentum"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for shared(un,u,v)
    for (SizeType i = 1; i <= (grid - 2); i++) {
        for (SizeType j = 1; j <= (grid - 1); j++) {
            un[(i) * (grid + 1) + j] = u[(i) * (grid + 1) + j]
                - dt
                    * ((u[(i + 1) * (grid + 1) + j] * u[(i + 1) * (grid + 1) + j] - u[(i - 1) * (grid + 1) + j] * u[(i - 1) * (grid + 1) + j]) / 2.0 / dx
                    + 0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*(grid + 1) + j] + v[(i + 1) * (grid + 1) + j])
                            - (u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) * (v[(i + 1) * (grid + 1) + j - 1] + v[(i)*(grid + 1) + j - 1])) / dy)
                    - dt / dx * (p[(i + 1) * (grid + 1) + j] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
                    * ((u[(i + 1) * (grid + 1) + j] - 2.0 * u[(i) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j]) / dx / dx
                     + (u[(i) * (grid + 1) + j + 1] - 2.0 * u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) / dy / dy);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["solveUMomentum"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["solveUMomentum"] =   (grid-2)*(grid-1)*((4)+(4)+(4)+(4)); // write un read u v p
}

void Simulator::applyBoundaryU() {
    countMap["applyBoundaryU"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType j = 1; j <= (grid - 1); j++) {
        un[(0) * (grid + 1) + j] = 0.0;
        un[(grid - 1) * (grid + 1) + j] = 0.0;
    }

    for (SizeType i = 0; i <= (grid - 1); i++) {
        un[(i) * (grid + 1) + 0] = -un[(i) * (grid + 1) + 1];
        un[(i) * (grid + 1) + grid] = 2 - un[(i) * (grid + 1) + grid - 1];
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["applyBoundaryU"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["applyBoundaryU"] =   2*(grid-1)*((4)+(4)); // write un two times and write un two times
}

void Simulator::solveVMomentum(const FloatType Re) {
    #pragma omp parallel for collapse(2)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        for (SizeType j = 1; j <= (grid - 2); j++) {
            vn[(i)*(grid + 1) + j] = v[(i)*(grid + 1) + j]
                - dt * (0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*(grid + 1) + j] + v[(i + 1) * (grid + 1) + j])
                              - (u[(i - 1) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j + 1]) * (v[(i)*(grid + 1) + j] + v[(i - 1) * (grid + 1) + j])) / dx
                              + (v[(i)*(grid + 1) + j + 1] * v[(i)*(grid + 1) + j + 1] - v[(i)*(grid + 1) + j - 1] * v[(i)*(grid + 1) + j - 1]) / 2.0 / dy)
                              - dt / dy * (p[(i) * (grid + 1) + j + 1] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
                              * ((v[(i + 1) * (grid + 1) + j] - 2.0 * v[(i)*(grid + 1) + j] + v[(i - 1) * (grid + 1) + j]) / dx / dx
                              + (v[(i)*(grid + 1) + j + 1] - 2.0 * v[(i)*(grid + 1) + j] + v[(i)*(grid + 1) + j - 1]) / dy / dy);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["solveVMomentum"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["solveVMomentum"] =   (grid-2)*(grid-1)*((4)+(4)+(4)+(4)); // write vn read u v p
}

void Simulator::applyBoundaryV() {
    countMap["applyBoundaryV"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType j = 1; j <= (grid - 2); j++) {
        vn[(0) * (grid + 1) + j] = -vn[(1) * (grid + 1) + j];
        vn[(grid)*(grid + 1) + j] = -vn[(grid - 1) * (grid + 1) + j];
    }

    for (SizeType i = 0; i <= (grid); i++) {
        vn[(i)*(grid + 1) + 0] = 0.0;
        vn[(i)*(grid + 1) + grid - 1] = 0.0;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["applyBoundaryV"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["applyBoundaryV"] =   (grid-2)*((4)+(4)) + (grid)*((4)+(4)); // write vn two times and write vn two times
}

void Simulator::solveContinuityEquationP(const FloatType delta) {
    countMap["solveContinuityEquationP"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for shared(p,pn,un,vn)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        for (SizeType j = 1; j <= (grid - 1); j++) {
            pn[(i) * (grid + 1) + j] = p[(i) * (grid + 1) + j]
                - dt * delta * ((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*(grid + 1) + j] - vn[(i)*(grid + 1) + j - 1]) / dy);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["solveContinuityEquationP"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["solveContinuityEquationP"] =   (grid-1)*(grid-1)*((4)+(4)+(4)+(4)); // write pn read un vb p
}

void Simulator::applyBoundaryP() {
    countMap["applyBoundaryP"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 1; i <= (grid - 1); i++) {
        pn[(i) * (grid + 1) + 0] = pn[(i) * (grid + 1) + 1];
        pn[(i) * (grid + 1) + grid] = pn[(i) * (grid + 1) + grid - 1];
    }

    #pragma omp parallel for
    for (SizeType j = 0; j <= (grid); j++) {
        pn[(0) * (grid + 1) + j] = pn[(1) * (grid + 1) + j];
        pn[(grid) * (grid + 1) + j] = pn[(grid - 1) * (grid + 1) + j];
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["applyBoundaryP"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["applyBoundaryP"] =   (grid-1)*((4)+(4)) + (grid)*((4)+(4)); // write vn two times and write vn two times
}

Simulator::FloatType Simulator::calculateError() {
    countMap["calculateError"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    FloatType error = 0.0;

    #pragma omp parallel for reduction(+:error)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        for (SizeType j = 1; j <= (grid - 1); j++) {
            m[(i) * (grid + 1) + j] =
                ((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*(grid + 1) + j] - vn[(i)*(grid + 1) + j - 1]) / dy);
            error += fabs(m[(i) * (grid + 1) + j]);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["calculateError"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["calculateError"] =   (grid-1)*(grid-1)*((4)+(4)+(4)+(4)); // write m read un vn and write error

    return error;
}

void Simulator::iterateU() {
    countMap["iterateU"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::swap(u, un);
    // for (SizeType i = 0; i <= (grid - 1); i++) {
    //     for (SizeType j = 0; j <= (grid); j++) {
    //         u[(i) * (grid + 1) + j] = un[(i) * (grid + 1) + j];
    //     }
    // }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["iterateU"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["iterateU"] =   (grid-1)*(grid)*((4)+(4)); // write u read un
}

void Simulator::iterateV() {
    countMap["iterateV"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::swap(v, vn);
    // for (SizeType i = 0; i <= (grid); i++) {
    //     for (SizeType j = 0; j <= (grid - 1); j++) {
    //         v[(i)*(grid + 1) + j] = vn[(i)*(grid + 1) + j];
    //     }
    // }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["iterateV"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["iterateV"] =   (grid-1)*(grid)*((4)+(4)); // write v read vn
}

void Simulator::iterateP() {
    countMap["iterateP"] += 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::swap(p, pn);
    // for (SizeType i = 0; i <= (grid); i++) {
    //     for (SizeType j = 0; j <= (grid); j++) {
    //         p[(i) * (grid + 1) + j] = pn[(i) * (grid + 1) + j];
    //     }
    // }
    auto t2 = std::chrono::high_resolution_clock::now();
    timeMap["iterateP"] = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    bytesMovMap["iterateP"] =   (grid)*(grid)*((4)+(4)); // write p read pn
}

void Simulator::deallocate() {
    // it doesn't do anything until we use vectors
    // because that deallocates automatically
    // but if we have to use a more raw data structure later it is needed
    // and when the the Tests overwrites some member those might won't deallocate
}

Simulator::Simulator(SizeType gridP)
    : grid([](auto g) {
          if (g <= 1) {
              throw std::runtime_error("Grid is smaller or equal to 1.0, give larger number");
          }
          return g;
      }(gridP)),
      dx(1.0 / static_cast<FloatType>(grid - 1)),
      dy(1.0 / static_cast<FloatType>(grid - 1)),
      dt(0.001 / std::pow(grid / 128.0 * 2.0, 2.0)),
      u((grid + 1) * (grid + 1)),
      un((grid + 1) * (grid + 1)),
      v((grid + 1) * (grid + 1)),
      vn((grid + 1) * (grid + 1)),
      p((grid + 1) * (grid + 1)),
      pn((grid + 1) * (grid + 1)),
      m((grid + 1) * (grid + 1)) {
    MPIHandler::getInstance()->handleMPIResource();
    initU();
    initV();
    initP();
}

void Simulator::run(const FloatType delta, const FloatType Re, unsigned maxSteps) {
    if (printing) {
        fmt::print("Running simulation with delta: {}, Re: {}\n", delta, Re);
    }
    auto error = std::numeric_limits<FloatType>::max();
    unsigned step = 1;
    while (error > 0.00000001 && step <= maxSteps) {
        solveUMomentum(Re);
        applyBoundaryU();

        solveVMomentum(Re);
        applyBoundaryV();

        solveContinuityEquationP(delta);
        applyBoundaryP();

        error = calculateError();

        if (printing && (step % 1000 == 1)) {
            fmt::print("Error is {} for the step {}\n", error, step);
        }

        iterateU();
        iterateV();
        iterateP();
        ++step;
    }
    std::cout
    << std::setw(25)
    <<"Name"
    << std::setw(10)
    <<"Count"
    << std::setw(10)
    <<"Time(ms)"
    << std::setw(10)
    <<"GB/s"
    << std::endl;
    std::cout <<"--------------------------------------------------------------" << std::endl;
    for (size_t i = 0; i < countMap.size() ; i++)
    {
        std::cout
        << std::setw(25) << functionNames[i]
        << " "
        << std::setw(10) << countMap[functionNames[i]]
        << " "
        << std::setw(10) << double(timeMap[functionNames[i]])/1e3
        << " ";
        if(timeMap[functionNames[i]] != 0){
            std::cout << std::setw(10) << double(bytesMovMap[functionNames[i]])/1e9 / double(timeMap[functionNames[i]])/1e6;
        }
        else{
            std::cout << std::setw(10) << "-";
        }

        std::cout << std::endl;
    }
    std::cout <<"--------------------------------------------------------------"  << std::endl;
}

Simulator::~Simulator() { deallocate(); }
