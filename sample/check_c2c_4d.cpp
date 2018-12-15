/*******************************************************
*    This file is test for FFT c2c 4D interface
*       based on the OpenFFT sample code: check_c2c_4d.c
*
*   OpenFFT library
*   http://www.openmx-square.org/openfft/
******************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "openfft.hpp"

#define RANGE 4.5


template <class T>
void check_4d_array(const int n1, const int n2, const int n3, const int n4,
                    const T *array_3d,
                    const T *ref_arr  ){

    int fail_count = 0;
    for(int iw=0; iw<n1; ++iw){
        for(int iz=0; iz<n2; ++iz){
            for(int iy=0; iy<n3; ++iy){
                for(int ix=0; ix<n4; ++ix){
                    const int pos = iw*(n4*n3*n2)
                                  + iz*(n4*n3)
                                  + iy*(n4)
                                  + ix;

                    const auto elem = array_3d[pos];
                    const auto ref  = ref_arr[pos];
                    if( std::abs(elem.r - ref.r) > 0.001 ||
                        std::abs(elem.i - ref.i) > 0.001   ){

                        printf("ERROR array[%d,%d,%d,%d] data=(% 3.3f,% 3.3f), ref=(% 3.3f,% 3.3f)\n",
                                iw, iz, iy, ix, elem.r, elem.i, ref.r, ref.i);
                        ++fail_count;
                    }
                }
            }
        }
    }
    if(fail_count == 0){
        printf("   Check done. All elements are correct.\n");
    } else {
        printf("   Check done. Some elements are incorrect. failed = %d/%d points.\n",
               fail_count, n1*n2*n3*n4);
    }
}


int main(int argc, char* argv[])
{
    int numprocs,myid;
    int const N1=3,N2=4,N3=5,N4=6;
    dcomplex Input[N1][N2][N3][N4],Output[N1][N2][N3][N4];
    dcomplex Out[N1][N2][N3][N4],Output_ref[N1][N2][N3][N4];
    int offt_measure,measure_time,print_memory;
    int i,j,k,m;
    double factor;
    FILE *file_in;
    char BUF[1000];

    /* MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    if(!myid) printf("Executing with %d processes\n",numprocs);

    /* Set global input */

    file_in = fopen("check_c2c_4d.din","r");
    fgets(BUF,sizeof(BUF),file_in);
    fgets(BUF,sizeof(BUF),file_in);
    fgets(BUF,sizeof(BUF),file_in);
    fgets(BUF,sizeof(BUF),file_in);
    for(m=0;m<N1;m++){
        for(i=0;i<N2;i++){
            for(j=0;j<N3;j++){
                for(k=0;k<N4;k++){
                    fgets(BUF,sizeof(BUF),file_in);
                    sscanf(BUF,"%lf  %lf\n",&Input[m][i][j][k].r,&Input[m][i][j][k].i);
                }
            }
        }
    }
    fclose(file_in);

    /* Select auto-tuning of communication */

    if(argc==2){
        offt_measure = atoi(argv[1]);
    }
    else{
        offt_measure = 0;
    }

    /* Set whether to use the timing and print memory functions of OpenFFT
       or not. Default=0 (not use) */

    measure_time = 0;
    print_memory = 1;

    /* Initialize OpenFFT */
    OpenFFT::Manager<double> fft_mngr;
    fft_mngr.init_c2c_4d(N1, N2, N3, N4,
                         offt_measure, measure_time, print_memory);

    const auto My_Max_NumGrid = fft_mngr.get_max_n_grid();
    const auto My_NumGrid_In  = fft_mngr.get_n_grid_in();
    const auto My_Index_In    = fft_mngr.get_index_in();
    const auto My_NumGrid_Out = fft_mngr.get_n_grid_out();
    const auto My_Index_Out   = fft_mngr.get_index_out();

    /* Allocate local input and output arrays */

    std::vector<dcomplex> input_buffer, output_buffer;
    input_buffer.resize( My_NumGrid_In );
    output_buffer.resize(My_NumGrid_Out);

    for(auto& v : input_buffer) { v.r = 0.0; v.i = 0.0; }
    for(auto& v : output_buffer){ v.r = 0.0; v.i = 0.0; }

    /* Set local input */

    MPI_Barrier(MPI_COMM_WORLD);


    //------ copy 4D array data into local input buffer
    fft_mngr.copy_4d_array_into_input_buffer( &(Input[0][0][0][0]) , input_buffer);


    printf("myid=%4d: Input in the ABCD(XYZU) order with %d grid points",
             myid, My_NumGrid_In);
    if(My_NumGrid_In > 0){
        printf("from (A=%d,B=%d,C=%d,D=%d) to (A=%d,B=%d,C=%d,D=%d)\n",
                My_Index_In[0], My_Index_In[1], My_Index_In[2], My_Index_In[3],
                My_Index_In[4], My_Index_In[5], My_Index_In[6], My_Index_In[7]);
    }
    else{
        printf("\n");
    }

    /* FFT transform */

    fft_mngr.fft_c2c_4d_forward( input_buffer.data(), output_buffer.data() );

    /* Get local output */

    MPI_Barrier(MPI_COMM_WORLD);

    factor = sqrt(N1*N2*N3*N4);

    for(m=0;m<N1;m++){
        for(i=0;i<N2;i++){
            for(j=0;j<N3;j++){
                for(k=0;k<N4;k++){
                    Out[m][i][j][k].r = 0.0;
                    Out[m][i][j][k].i = 0.0;
                    Output[m][i][j][k].r = 0.0;
                    Output[m][i][j][k].i = 0.0;
                }
            }
        }
    }


    //------ copy local output buffer into 4D array data
    fft_mngr.copy_4d_array_from_output_buffer( &(Out[0][0][0][0]) , output_buffer);

    for(m=0;m<N1;m++){
        for(i=0;i<N2;i++){
            for(j=0;j<N3;j++){
                for(k=0;k<N4;k++){
                    Out[m][i][j][k].r /= factor;
                    Out[m][i][j][k].i /= factor;
                }
            }
        }
    }

    printf("myid=%4d: Output in the DCBA(UZYX) order with %d grid points",
            myid, My_NumGrid_Out);
    if(My_NumGrid_Out > 0){
        printf("from (D=%d,C=%d,B=%d,A=%d) to (D=%d,C=%d,B=%d,A=%d)\n",
                My_Index_Out[0], My_Index_Out[1], My_Index_Out[2], My_Index_Out[3],
                My_Index_Out[4], My_Index_Out[5], My_Index_Out[6], My_Index_Out[7] );
    }
    else{
        printf("\n");
    }

    /* Gather results from all processes */

    MPI_Allreduce(Out, Output, N1*N2*N3*N4,
                  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    /* Print global output */

    if(!myid){
        sprintf(BUF,"check_c2c_4dx%d.dout",numprocs);
        file_in = fopen(BUF,"w");
        for(m=0;m<N1;m++){
            for(i=0;i<N2;i++){
                for(j=0;j<N3;j++){
                    for(k=0;k<N4;k++){
                        fprintf(file_in,"%10.3f  %10.3f\n",
                        Output[m][i][j][k].r,Output[m][i][j][k].i);
                    }
                }
            }
        }
        fclose(file_in);
    }


    if(!myid){
        file_in = fopen("check_c2c_4d.dout","r");
        for(m=0;m<N1;m++){
            for(i=0;i<N2;i++){
                for(j=0;j<N3;j++){
                    for(k=0;k<N4;k++){
                        fgets(BUF,sizeof(BUF),file_in);
                        sscanf(BUF,"%lf  %lf\n",
                        &Output_ref[m][i][j][k].r,&Output_ref[m][i][j][k].i);
                    }
                }
            }
        }
        fclose(file_in);
    }

    /* Error check */

    if(!myid){
        printf("\n");
        printf(" --- check FFT output ( using copy_4d_array_from_output_buffer() & MPI_Allreduce() )\n");
        check_4d_array(N1, N2, N3, N4, &(Output[0][0][0][0]), &(Output_ref[0][0][0][0]) );
    }

    //------ copy local output buffer into 4D array data
    fft_mngr.allgather_4d_array( &(Output[0][0][0][0]) , output_buffer);

    for(m=0;m<N1;m++){
        for(i=0;i<N2;i++){
            for(j=0;j<N3;j++){
                for(k=0;k<N4;k++){
                    Output[m][i][j][k].r /= factor;
                    Output[m][i][j][k].i /= factor;
                }
            }
        }
    }

    if(!myid){
        printf("\n");
        printf(" --- check FFT output ( using Manager::allgather_4d_array() )\n");
        check_4d_array(N1, N2, N3, N4, &(Output[0][0][0][0]), &(Output_ref[0][0][0][0]) );
    }


    /* Finalize OpenFFT */

    fft_mngr.finalize();

    MPI_Finalize();

}
