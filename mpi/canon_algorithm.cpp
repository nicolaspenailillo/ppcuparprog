#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<fstream>

class Matrix
{
    public:
        void load(const std::string& matrixroute, std::string name);
        int row, col;
        float *matrix;
};

void Matrix::load(const std::string& matrixroute, std::string name)
{
    std::ifstream file(matrixroute);
    if(!file)
    {
        std::cerr << "Error opening matrix file.\n";
        return;
    }

    file >> row >> col;
    if(row < 1 || col < 1)
    {   std::cout << col << std::endl;
        std::cerr << "Matrix sizes are out of bounds.\n";
        return;
    }

    matrix=(float*)calloc(sizeof(float),row*col);

    // Read the input file.
    std:: cout << name << " Matrix:"<< std::endl;
    for (size_t i = 0; i < row; i++)
    {
        for (size_t j = 0; j < col; j++)
        {
            file >> matrix[i+j];
            std:: cout << matrix[i+j] <<" ";
        }
        std::cout << std::endl;
    }
}

void printMatrix(std::string name, int row, int col, float* matrix){
    // Read the input file.
    std:: cout << name << " Matrix:"<< std::endl;
    for (size_t i = 0; i < row; i++)
    {
        for (size_t j = 0; j < col; j++)
        {
            std:: cout << matrix[i+j] <<" ";
        }
        std::cout << std::endl;
    }
}


int main(int argc,char *argv[])
{
   int rc, rank, len, numtasks;
   char hostname[MPI_MAX_PROCESSOR_NAME];
   float *C , a=0,b=0,c=0,n;
   Matrix A;
   Matrix B;


   rc = MPI_Init(&argc, &argv);
   if (rc != MPI_SUCCESS)
   {
       std::cout << "Error starting MPI program. Terminating.\n";
       MPI_Abort(MPI_COMM_WORLD, rc);
   }

   MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Get_processor_name(hostname, &len);
   if(rank==0)
   {
    A.load("A.txt","A");
    B.load("A.txt","B");
   }

   MPI_Bcast(&A.row,1,MPI_INT,0,MPI_COMM_WORLD);
   int periods[]={1,1};
   int dims[]={A.row,A.row};
   int coords[2];
   int right=0, left=0, down=0, up=0;
   MPI_Comm cart_comm;
   MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,1,&cart_comm );
   MPI_Scatter(A.matrix,1,MPI_FLOAT,&a,1,MPI_FLOAT,0,cart_comm);
   MPI_Scatter(B.matrix,1,MPI_FLOAT,&b,1,MPI_FLOAT,0,cart_comm);
   MPI_Comm_rank(cart_comm,&rank);
   MPI_Cart_coords(cart_comm,rank,2,coords);
   MPI_Cart_shift(cart_comm, 1, coords[0], &left,&right);
   MPI_Cart_shift(cart_comm, 0, coords[1], &up,&down);
   MPI_Sendrecv_replace(&a,1,MPI_FLOAT,left,11,right,11,cart_comm,MPI_STATUS_IGNORE);
   MPI_Sendrecv_replace(&b,1,MPI_FLOAT,up,11,down,11,cart_comm,MPI_STATUS_IGNORE);
   c = c + a*b;
   for(int i=1;i<A.row;i++)
   {
     MPI_Cart_shift(cart_comm, 1, 1, &left,&right);
     MPI_Cart_shift(cart_comm, 0, 1, &up,&down);
     MPI_Sendrecv_replace(&a,1,MPI_FLOAT,left,11,right,11,cart_comm,MPI_STATUS_IGNORE);
     MPI_Sendrecv_replace(&b,1,MPI_FLOAT,up,11,down,11,cart_comm,MPI_STATUS_IGNORE);
     c = c + a*b;
   }

   C=(float*)calloc(sizeof(float),A.row*A.row);
   MPI_Gather(&c,1,MPI_FLOAT,C,1,MPI_FLOAT,0,cart_comm);


   if(rank==0)
   {
      printMatrix("C", A.row, A.col, C);
   }
   MPI_Finalize();
   return 0;
}