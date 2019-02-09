/*******************************************************
*    This file is test for FFT r2c 3D interface
*       based on the OpenFFT sample code: check_r2c_3d.c
*
*   OpenFFT library
*   http://www.openmx-square.org/openfft/
******************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "color_printer.hpp"

#include "openfft.hpp"

int main(int argc, char* argv[])
{
  int numprocs,myid;
  int const N1=2,N2=3,N3=4;
  int N3r=N3/2+1;
  int offt_measure,measure_time,print_memory;
  int i,j,k,l;
  double factor;

  double            Input[N1][N2][N3];
  OpenFFT::dcomplex Output[N1][N2][N3r];
  OpenFFT::dcomplex Output_ref[N1][N2][N3r];

  /* MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  /* Set global input */

  Input[0][0][0] = 1.000;
  Input[0][0][1] = 0.999;
  Input[0][0][2] = 0.987;
  Input[0][0][3] = 0.936;
  Input[0][1][0] = 0.994;
  Input[0][1][1] = 0.989;
  Input[0][1][2] = 0.963;
  Input[0][1][3] = 0.891;
  Input[0][2][0] = 0.903;
  Input[0][2][1] = 0.885;
  Input[0][2][2] = 0.823;
  Input[0][2][3] = 0.694;
  Input[1][0][0] = 0.500;
  Input[1][0][1] = 0.499;
  Input[1][0][2] = 0.487;
  Input[1][0][3] = 0.436;
  Input[1][1][0] = 0.494;
  Input[1][1][1] = 0.489;
  Input[1][1][2] = 0.463;
  Input[1][1][3] = 0.391;
  Input[1][2][0] = 0.403;
  Input[1][2][1] = 0.385;
  Input[1][2][2] = 0.323;
  Input[1][2][3] = 0.194;

  /* Select auto-tuning of communication */

  offt_measure = 0;

  /* Set whether to use the timing and print memory functions of OpenFFT
     or not. Default=0 (not use) */

  measure_time = 0;
  print_memory = 0;

  /* Initialize OpenFFT */
  OpenFFT::Manager<double> fft_mngr;
  fft_mngr.init_r2c_3d(N1, N2, N3,
                       offt_measure, measure_time, print_memory);
  const auto My_Max_NumGrid = fft_mngr.get_max_n_grid();
  const auto My_NumGrid_In  = fft_mngr.get_n_grid_in();
  const auto My_Index_In    = fft_mngr.get_index_in();
  const auto My_NumGrid_Out = fft_mngr.get_n_grid_out();
  const auto My_Index_Out   = fft_mngr.get_index_out();

  /* Allocate local input and output arrays */

  std::vector<double>            real_input_buffer;
  std::vector<OpenFFT::dcomplex> output_buffer;

  real_input_buffer.resize(My_Max_NumGrid);
  output_buffer.resize(My_Max_NumGrid);

  /* Set local input */

  MPI_Barrier(MPI_COMM_WORLD);

  printf("myid=%4d: Input in the ABC(XYZ) order with %d grid points ",
	 myid,My_NumGrid_In);
  if(My_NumGrid_In > 0){
    printf("from (A=%d,B=%d,C=%d) to (A=%d,B=%d,C=%d)\n",
	 My_Index_In[0],My_Index_In[1],My_Index_In[2],
	 My_Index_In[3],My_Index_In[4],My_Index_In[5]);
  }
  else{
    printf("\n");
  }

  for(i=0;i<My_Max_NumGrid;i++){
    real_input_buffer[i] = 0.0;
    output_buffer[i].r = 0.0;
    output_buffer[i].i = 0.0;
  }

  //------ copy 3D array data into local input buffer
  fft_mngr.copy_3d_array_into_input_buffer( &(Input[0][0][0]) , real_input_buffer);

  /* Print global input */

  if(!myid){
    printf("Input values\n\n");
    for(i=0;i<N1;i++){
      printf("Input(i,j,k) for i = %d\n\n",i);
      for(j=0;j<N2;j++){
	printf("Real\t");
	for(k=0;k<N3;k++){
	  printf("% 3.3f\t",Input[i][j][k]);
	}
	printf("\n\n");
      }
    }
  }


  /* FFT transform */

  //------ call exec fft through OpenFFT::Manager
  //          "buffer" argument accepts std::vector<OpenFFT::dcomplex> or pointer <OpenFFT::dcomplex*>.
  fft_mngr.fft_r2c_3d_forward(real_input_buffer, output_buffer);

  /* Get local output */

  MPI_Barrier(MPI_COMM_WORLD);
  printf("myid=%4d: Output in the CBA(ZYX) order with %d grid points ",
	 myid,My_NumGrid_Out);
  if(My_NumGrid_Out > 0){
    printf("from (C=%d,B=%d,A=%d) to (C=%d,B=%d,A=%d)\n",
	   My_Index_Out[0],My_Index_Out[1],My_Index_Out[2],
	 My_Index_Out[3],My_Index_Out[4],My_Index_Out[5]);
  }
  else{
    printf("\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  factor = sqrt(N1*N2*N3);

  for(i=0;i<N1;i++){
    for(j=0;j<N2;j++){
      for(k=0;k<N3r;k++){
	Output[i][j][k].r = 0.0;
	Output[i][j][k].i = 0.0;
      }
    }
  }

  /* Gather results from all processes */

  fft_mngr.allgather_3d_array( &(Output[0][0][0]), output_buffer);

  for(i=0;i<N1;i++){
    for(j=0;j<N2;j++){
      for(k=0;k<N3r;k++){
    	Output[i][j][k].r /= factor;
    	Output[i][j][k].i /= factor;
      }
    }
  }

  /* Print global output */

  if(!myid){
    printf("Output values\n\n");
    for(i=0;i<N1;i++){
      printf("Output(i,j,k) for i = %d\n\n",i);
      for(j=0;j<N2;j++){
	printf("Real\t");
	for(k=0;k<N3r;k++){
	  printf("% 3.3f\t",Output[i][j][k].r);
	}
	printf("\nImag\t");
	for(k=0;k<N3r;k++){
	  printf("% 3.3f\t",Output[i][j][k].i);
	}
	printf("\n\n");
      }
    }
  }

  /* Error check */

  Output_ref[0][0][0].r = 3.292; Output_ref[0][0][0].i = 0.000;
  Output_ref[0][0][1].r = 0.051; Output_ref[0][0][1].i =-0.144;
  Output_ref[0][0][2].r = 0.113; Output_ref[0][0][2].i = 0.000;
  Output_ref[0][1][0].r = 0.143; Output_ref[0][1][0].i =-0.188;
  Output_ref[0][1][1].r = 0.016; Output_ref[0][1][1].i = 0.051;
  Output_ref[0][1][2].r =-0.024; Output_ref[0][1][2].i = 0.025;
  Output_ref[0][2][0].r = 0.143; Output_ref[0][2][0].i = 0.188;
  Output_ref[0][2][1].r =-0.050; Output_ref[0][2][1].i = 0.016;
  Output_ref[0][2][2].r =-0.024; Output_ref[0][2][2].i =-0.025;
  Output_ref[1][0][0].r = 1.225; Output_ref[1][0][0].i = 0.000;
  Output_ref[1][0][1].r = 0.000; Output_ref[1][0][1].i = 0.000;
  Output_ref[1][0][2].r = 0.000; Output_ref[1][0][2].i = 0.000;
  Output_ref[1][1][0].r = 0.000; Output_ref[1][1][0].i = 0.000;
  Output_ref[1][1][1].r = 0.000; Output_ref[1][1][1].i = 0.000;
  Output_ref[1][1][2].r = 0.000; Output_ref[1][1][2].i = 0.000;
  Output_ref[1][2][0].r = 0.000; Output_ref[1][2][0].i =-0.000;
  Output_ref[1][2][1].r =-0.000; Output_ref[1][2][1].i = 0.000;
  Output_ref[1][2][2].r = 0.000; Output_ref[1][2][2].i =-0.000;

  l = 0;
  if(!myid){
    for(i=0;i<N1;i++){
      for(j=0;j<N2;j++){
	for(k=0;k<N3r;k++){
        const auto value = Output[i][j][k];
        const auto ref   = Output_ref[i][j][k];
	  if(fabs(value.r - ref.r)>0.001 ||
	     fabs(value.i - ref.i)>0.001){
	    l = 1;
        print_yellow("ERROR");
	    printf(" Output[%d,%d,%d]  value=(% 3.3f,% 3.3f), ref=(% 3.3f,% 3.3f)\n",
		       i,j,k, value.r, value.i, ref.r, ref.i);
	  }
	}
      }
    }
    if (l==0){
      print_green("Check done.");
      printf(" All output elements are correct.\n");
    } else {
      print_red("CHeck failure.");
      printf(" Some output elements are incorrect.\n");
    }
  }

  /* Finalize OpenFFT */

  fft_mngr.finalize();

  MPI_Finalize();

}
