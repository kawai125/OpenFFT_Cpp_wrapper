/*******************************************************
*    This file is test for FFT c2c 4D interface
*       based on the OpenFFT sample code: check_c2c_4d.c
*
*   OpenFFT library
*   http://www.openmx-square.org/openfft/
******************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "openfft.hpp"

#include "color_printer.hpp"
#include "test_tool.hpp"

#define RANGE 4.5


int main(int argc, char* argv[])
{
    int numprocs,myid;
    const int N1=3;
    const int N2=4;
    const int N3=5;
    const int N4=6;
    const int n_total = N1*N2*N3*N4;
    int offt_measure, measure_time, print_memory;
    int i,j,k,m;
    double factor;


    OpenFFT::dcomplex Input[N1][N2][N3][N4];
    OpenFFT::dcomplex Out[N1][N2][N3][N4];
    OpenFFT::dcomplex Output[N1][N2][N3][N4];
    OpenFFT::dcomplex Output_ref[N1][N2][N3][N4];

    /* MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    if(!myid) printf("Executing with %d processes\n",numprocs);

    /* Set global input */

    if( myid == 0){
        char BUF[100];
        FILE *file_in = fopen("check_c2c_4d.din","r");
        if( file_in == nullptr ) throw std::logic_error("file not found.");

        int m1, m2, m3, m4;
        fgets(BUF,sizeof(BUF),file_in); sscanf(BUF, "%d\n", &m1);
        fgets(BUF,sizeof(BUF),file_in); sscanf(BUF, "%d\n", &m2);
        fgets(BUF,sizeof(BUF),file_in); sscanf(BUF, "%d\n", &m3);
        fgets(BUF,sizeof(BUF),file_in); sscanf(BUF, "%d\n", &m4);

        if(m1 != N1 || m2 != N2 || m3 != N3 || m4 != N4){
            throw std::logic_error("the file format was differ.");
        }

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
    }
    MPI_Bcast( &(Input[0][0][0][0]), n_total, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD );

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

    std::vector<OpenFFT::dcomplex> input_buffer, output_buffer;
    input_buffer.reserve( My_Max_NumGrid);
    output_buffer.reserve(My_Max_NumGrid);
    input_buffer.resize( My_NumGrid_In );
    output_buffer.resize(My_NumGrid_Out);

    for(auto& v : input_buffer) { v.r = 0.0; v.i = 0.0; }
    for(auto& v : output_buffer){ v.r = 0.0; v.i = 0.0; }

    /* Set local input */

    MPI_Barrier(MPI_COMM_WORLD);


    //------ copy 4D array data into local input buffer
    fft_mngr.copy_4d_array_into_input_buffer( &(Input[0][0][0][0]) , input_buffer);


    MPI_Barrier(MPI_COMM_WORLD);
    for(int i_proc=0; i_proc<numprocs; ++i_proc){
        if(i_proc == myid){
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
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* FFT transform */

    fft_mngr.fft_c2c_4d_forward( input_buffer, output_buffer );

    /* Get local output */

    MPI_Barrier(MPI_COMM_WORLD);

    factor = sqrt(n_total);

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

    MPI_Allreduce( &(Out[0][0][0][0]), &(Output[0][0][0][0]), n_total,
                  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    /* Print global output */

    if(myid == 0){
        char BUF[100];
        sprintf(BUF,"check_c2c_4dx%d.dout",numprocs);
        FILE *file_out = fopen(BUF,"w");
        if( file_out == nullptr ) throw std::logic_error("failure to create file.");
        for(m=0;m<N1;m++){
            for(i=0;i<N2;i++){
                for(j=0;j<N3;j++){
                    for(k=0;k<N4;k++){
                        fprintf(file_out,"%10.3f  %10.3f\n",
                        Output[m][i][j][k].r,Output[m][i][j][k].i);
                    }
                }
            }
        }
        fclose(file_out);
    }

    if(myid == 0){
        char BUF[100];
        FILE *file_in = fopen("check_c2c_4d.dout","r");
        if( file_in == nullptr ) throw std::logic_error("file not found.");
        for(m=0;m<N1;m++){
            for(i=0;i<N2;i++){
                for(j=0;j<N3;j++){
                    for(k=0;k<N4;k++){
                        fgets(BUF,sizeof(BUF),file_in);
                        sscanf(BUF,"%lf  %lf\n",
                                   &Output_ref[m][i][j][k].r, &Output_ref[m][i][j][k].i);
                    }
                }
            }
        }
        fclose(file_in);
    }
    MPI_Bcast( &(Output_ref[0][0][0][0]), n_total, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);


    //==========================================================================
    //  Gather results interface test
    //==========================================================================
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            printf(     "\n");
            print_green("[checking FFT output]\n");
            printf(     "\n");
            print_green(" -- using copy_4d_array_from_output_buffer() & MPI_Allreduce()\n");
            TEST::check_4d_array(N1, N2, N3, N4,
                                 &(Output[0][0][0][0]), &(Output_ref[0][0][0][0]) );
        }

        if(myid == 0) printf(     "\n");
        for(int i_proc=0; i_proc<numprocs; ++i_proc){

            fft_mngr.gather_4d_array( &(Output[0][0][0][0]), output_buffer, i_proc );

            MPI_Barrier(MPI_COMM_WORLD);
            if(myid == i_proc){
                print_green(" -- using Manager::gather_4d_array()");
                printf(     " at proc=%d\n", i_proc);

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
                TEST::check_4d_array(N1, N2, N3, N4,
                                     &(Output[0][0][0][0]), &(Output_ref[0][0][0][0]) );
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }


        fft_mngr.allgather_4d_array( &(Output[0][0][0][0]), output_buffer );

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

        if(myid == 0) printf(     "\n");
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i_proc=0; i_proc<numprocs; ++i_proc){
            if(myid == i_proc){
                print_green(" -- using Manager::allgather_4d_array()");
                printf(     " at proc=%d\n", i_proc);
                TEST::check_4d_array(N1, N2, N3, N4,
                                     &(Output[0][0][0][0]), &(Output_ref[0][0][0][0]) );
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }


    /* Finalize OpenFFT */

    fft_mngr.finalize();

    MPI_Finalize();

}
