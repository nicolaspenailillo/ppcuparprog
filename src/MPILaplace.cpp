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


void MPISetup(unsigned *xmax, unsigned *ymax) {
  //
  //get number of processes
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  //Figure out process layout
  int periods[2]={0,0}, reorder=0;
  dims[0] = dims[1] = 0;
  MPI_Dims_create(nprocs, 2, dims);
  //Create cartesian communicator for 2D, dims[0]*dims[1] processes,
  //without periodicity and reordering
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);
  //Get my rank in the new communicator
  MPI_Comm_rank(cartcomm, &my_rank);
  //Get my coordinates coords[0] and coords[1]
  MPI_Cart_coords(cartcomm, my_rank, 2, coords);
  //Get my neighbours in dimension 0
  MPI_Cart_shift(cartcomm, 0, 1, &prev_x, &next_x);
  //Get my neighbours in dirmension 1
  MPI_Cart_shift(cartcomm, 1, 1, &prev_y, &next_y);
  //Save full sizes in x and y directions
  xmax_full = *xmax;
  ymax_full = *ymax;
  //Figure out where my domain begins
  gbl_x_begin = coords[0]*(*xmax)/dims[0];
  gbl_y_begin = coords[1]*(*ymax)/dims[1];
  //Modify xmax and ymax and account for rounding issues
  if (coords[0] == dims[0]-1) (*xmax) = (*xmax) - (dims[0]-1)*((*xmax)/dims[0]);
  else (*xmax) = (*xmax)/dims[0];
  if (coords[1] == dims[0]-1) (*ymax) = (*ymax) - (dims[1]-1)*((*ymax)/dims[1]);
  else (*ymax) = (*ymax)/dims[1];
  //Let's set MPI Datatypes
  MPI_Type_vector((*ymax)+2,1,(*xmax)+2, MPI_DOUBLE, &vertSlice);
  MPI_Type_vector((*xmax)+2,1,1, MPI_DOUBLE, &horizSlice);
  MPI_Type_commit(&vertSlice);
  MPI_Type_commit(&horizSlice);

}


void exchangeHalo(unsigned xmax,  unsigned ymax, double *arr) {
  //send/receive vertical slices to previous and next neighbours in X direction
  MPI_Sendrecv(&arr[1]     ,1,vertSlice,prev_x ,0,
      &arr[xmax+1],1,vertSlice,next_x,0,
      MPI_COMM_WORLD,MPI_STATUS_IGNORE);

  MPI_Sendrecv(&arr[xmax],1,vertSlice,next_x,0,
      &arr[0]   ,1,vertSlice,prev_x ,0,
      MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  //send/receive vertical slices to previous and next neighbours in Y direction
  MPI_Sendrecv(&arr[ymax*(xmax+2)]  ,1,horizSlice,next_y,0,
      &arr[0]     ,1,horizSlice,prev_y,0,
      MPI_COMM_WORLD,MPI_STATUS_IGNORE);

  MPI_Sendrecv(&arr[1*(xmax+2)]      ,1,horizSlice,prev_y,0,
      &arr[(ymax+1)*(xmax+2)] ,1,horizSlice,next_y,0,
      MPI_COMM_WORLD,MPI_STATUS_IGNORE);

}
