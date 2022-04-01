#include <mpi.h>

extern int my_rank;
extern int prev_y;
extern int next_y;
extern int next_x;
extern int prev_x;

extern int xmax_full;
extern int ymax_full;

extern int gbl_x_begin;
extern int gbl_y_begin;

void MPISetup(unsigned *xmax, unsigned *ymax);
void exchangeHalo(unsigned xmax, unsigned ymax, double *arr);