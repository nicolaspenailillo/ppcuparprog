#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "MPILaplace.hpp"

int my_rank;
int nprocs;
int dims[2];
int coords[2];
int prev_y;
int next_y;
int next_x;
int prev_x;
MPI_Datatype vertSlice, horizSlice;
int xmax_full;
int ymax_full;
int gbl_x_begin;
int gbl_y_begin;
MPI_Comm cartcomm;

// please remove the [[maybe_unused]]s and this comment
void MPISetup([[maybe_unused]] unsigned *xmax, [[maybe_unused]] unsigned *ymax) {
  // 
  //get number of processes

  //Figure out process layout
  
  //Create cartesian communicator for 2D, dims[0]*dims[1] processes,
  //without periodicity and reordering
  
  //Get my rank in the new communicator

  //Get my coordinates coords[0] and coords[1]

  //Get my neighbours in dimension 0

  //Get my neighbours in dirmension 1

  //Save full sizes in x and y directions

  //Figure out where my domain begins

  //Modify xmax and ymax and account for rounding issues

  //Let's set MPI Datatypes

}

// please remove the [[maybe_unused]]s and this comment
void exchangeHalo([[maybe_unused]] unsigned xmax, [[maybe_unused]] unsigned ymax, [[maybe_unused]] double *arr) {
  //send/receive vertical slices to previous and next neighbours in X direction

  //send/receive vertical slices to previous and next neighbours in Y direction

}
