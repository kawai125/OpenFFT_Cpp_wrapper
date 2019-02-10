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

#include "openfft.hpp"

#include "color_printer.hpp"
#include "test_tool.hpp"



int main(int argc, char* argv[])
{
    int numprocs,myid;
    int const N1=2,N2=3,N3=4;
    int N3r=N3/2+1;
    int offt_measure,measure_time,print_memory;
    int i,j,k;
    double factor;

    double            Input[N1][N2][N3];
    OpenFFT::dcomplex Out[N1][N2][N3r];
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

    if(myid == 0){
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
                Out[i][j][k].r = 0.0;
                Out[i][j][k].i = 0.0;
            }
        }
    }

    /* Gather results from all processes */
    fft_mngr.copy_3d_array_from_output_buffer( &(Out[0][0][0]), output_buffer);

    for(i=0;i<N1;i++){
        for(j=0;j<N2;j++){
            for(k=0;k<N3r;k++){
                Out[i][j][k].r /= factor;
                Out[i][j][k].i /= factor;
            }
        }
    }

    for(i=0;i<N1;i++){
        for(j=0;j<N2;j++){
            for(k=0;k<N3r;k++){
                Output[i][j][k].r = 0.0;
                Output[i][j][k].i = 0.0;
            }
        }
    }

    MPI_Allreduce(Out, Output, N1*N2*N3r,
                  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    /* Print global output */

    if(myid == 0){
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

    //==========================================================================
    //  Gather results interface test
    //==========================================================================
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            printf(     "\n");
            print_green("[checking FFT output]\n");
            printf(     "\n");
            print_green(" -- using copy_3d_array_from_output_buffer() & MPI_Allreduce()\n");
            TEST::check_3d_array(N1, N2, N3r,
                                 &(Output[0][0][0]), &(Output_ref[0][0][0]) );
            printf(     "\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        for(int i_proc=0; i_proc<numprocs; ++i_proc){
            fft_mngr.gather_3d_array( &(Output[0][0][0]), output_buffer, i_proc );

            MPI_Barrier(MPI_COMM_WORLD);
            if(myid == i_proc){
                print_green(" -- using Manager::gather_3d_array()");
                printf(     " at proc=%d\n", i_proc);

                for(i=0;i<N1;i++){
                    for(j=0;j<N2;j++){
                        for(k=0;k<N3r;k++){
                            Output[i][j][k].r /= factor;
                            Output[i][j][k].i /= factor;
                        }
                    }
                }

                TEST::check_3d_array(N1, N2, N3r,
                                     &(Output[0][0][0]), &(Output_ref[0][0][0]) );
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }


        if(myid == 0){
            printf(     "\n");
            print_green(" -- using Manager::allgather_3d_array() )\n");
        }

        fft_mngr.allgather_3d_array( &(Output[0][0][0]), output_buffer );

        for(i=0;i<N1;i++){
            for(j=0;j<N2;j++){
                for(k=0;k<N3r;k++){
                    Output[i][j][k].r /= factor;
                    Output[i][j][k].i /= factor;
                }
            }
        }

        if(myid == 0){
            TEST::check_3d_array(N1, N2, N3r,
                                 &(Output[0][0][0]), &(Output_ref[0][0][0]) );
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //==========================================================================
    //  functor interface test
    //==========================================================================
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            printf("\n");
            print_green("[functor interface test]\n");
            printf("\n");
            print_green(" -- check functor interface for input buffer\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        struct ReduceValue_R{
            double v      = 0.0;
            double factor = 1.0;
            void operator () (const double arr_v, const double buf_v){
                this->v += std::abs(arr_v * buf_v / this->factor);
            }
        };
        ReduceValue_R reduce_value_r;

        reduce_value_r.factor = factor;
        fft_mngr.copy_3d_array_into_input_buffer( &(Input[0][0][0]), real_input_buffer);
        const auto local_sum_ib = fft_mngr.apply_3d_array_with_input_buffer( &(Input[0][0][0]),
                                                                              real_input_buffer,
                                                                              reduce_value_r );

        double sum_v_r;
        double sum_v_r_ref;
        MPI_Allreduce(&local_sum_ib.v, &sum_v_r, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        sum_v_r_ref = 0.0;
        for(i=0;i<N1;i++){
            for(j=0;j<N2;j++){
                for(k=0;k<N3;k++){
                     sum_v_r_ref += std::abs( Input[i][j][k] * Input[i][j][k] / factor);
                }
            }
        }

        if(myid == 0){
            bool check_flag = true;
            if( std::abs(sum_v_r - sum_v_r_ref) > 0.0001 ){
                check_flag = false;
                print_yellow("  ERROR:");
                printf(" reduced sum_v_r = % 6.5f, ref = % 6.5f\n",
                       sum_v_r, sum_v_r_ref);
            }

            if( ! check_flag ){
                print_red(  "        Check failure.");
                printf(     " Some elements are incorrect.\n");
            } else {
                print_green("        Check done.");
                printf(     " All elements are correct.\n");
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            printf("\n");
            print_green(" -- check functor interface for output buffer\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        struct ReduceValue_C{
            OpenFFT::dcomplex v {0.0, 0.0};
            double            factor = 1.0;
            void operator () (const OpenFFT::dcomplex arr_v, const OpenFFT::dcomplex buf_v){
                this->v.r += std::abs(arr_v.r * buf_v.r / this->factor);
                this->v.i += std::abs(arr_v.i * buf_v.i / this->factor);
            }
        };
        ReduceValue_C reduce_value_c;

        reduce_value_c.v.r    = 0.0;
        reduce_value_c.v.i    = 0.0;
        reduce_value_c.factor = factor;
        const auto local_sum_ob = fft_mngr.apply_3d_array_with_output_buffer( &(Output[0][0][0]),
                                                                              output_buffer,
                                                                              reduce_value_c );

        OpenFFT::dcomplex sum_v_c;
        OpenFFT::dcomplex sum_v_c_ref;
        MPI_Allreduce(&local_sum_ob.v, &sum_v_c, 1,
                      MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

        sum_v_c_ref.r = 0.0;
        sum_v_c_ref.i = 0.0;
        for(i=0;i<N1;i++){
            for(j=0;j<N2;j++){
                for(k=0;k<N3r;k++){
                    sum_v_c_ref.r += std::abs( Output[i][j][k].r * Output[i][j][k].r);
                    sum_v_c_ref.i += std::abs( Output[i][j][k].i * Output[i][j][k].i);
                }
            }
        }

        if(myid == 0){
            bool check_flag = true;
            if( std::abs(sum_v_c.r - sum_v_c_ref.r) > 0.0001 ||
                std::abs(sum_v_c.i - sum_v_c_ref.i) > 0.0001   ){
                check_flag = false;
                print_yellow("  ERROR:");
                printf(" reduced sum_v = (% 6.5f,% 6.5f), ref = (% 6.5f,% 6.5f)\n",
                       sum_v_c.r, sum_v_c.i, sum_v_c_ref.r, sum_v_c_ref.i);
            }

            if( ! check_flag ){
                print_red(  "        Check failure.");
                printf(     " Some elements are incorrect.\n");
            } else {
                print_green("        Check done.");
                printf(     " All elements are correct.\n");
            }
        }
    }

    //==========================================================================
    //  index sequence generator test
    //==========================================================================
    {
        std::vector<std::array<int, 3>> index_seq;

        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            printf("\n");
            print_green("[index sequence generator test]\n");
            printf("\n");
        }
        for(int i_proc=0; i_proc<numprocs; ++i_proc){
            if(i_proc == myid){
                std::ostringstream oss;
                oss_green(oss, " -- check index for input buffer");
                oss << " at proc=" << myid << "\n";
                printf(oss.str().c_str());

                std::vector<double> buf, buf_ref;
                fft_mngr.copy_3d_array_into_input_buffer( &(Input[0][0][0]), buf_ref);

                print_green("    Manager<>::gen_3d_input_index_sequence()\n");
                fft_mngr.gen_3d_input_index_sequence(index_seq);
                buf.clear();
                for(const auto& index : index_seq){
                    buf.push_back( Input[ index[0] ][ index[1] ][ index[2] ] );
                }
                TEST::check_buffer(My_NumGrid_In,
                                   buf.data(),
                                   buf_ref.data(), myid );

                for(int tgt_proc=0; tgt_proc<numprocs; ++tgt_proc){
                    const int n_grid_in = fft_mngr.get_n_grid_in(tgt_proc);
                    buf_ref.resize( n_grid_in );
                    fft_mngr.apply_3d_array_with_input_buffer( &(Input[0][0][0]), buf_ref, OpenFFT::CopyIntoBuffer{}, tgt_proc);

                    print_green("    Manager<>::gen_3d_input_index_sequence( tgt_proc )");
                    printf(", tgt_proc=%d\n", tgt_proc);

                    fft_mngr.gen_3d_input_index_sequence(index_seq, tgt_proc);
                    buf.clear();
                    for(const auto& index : index_seq){
                        buf.push_back( Input[ index[0] ][ index[1] ][ index[2] ] );
                    }
                    TEST::check_buffer(n_grid_in,
                                       buf.data(),
                                       buf_ref.data(), myid );
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0) printf("\n");
        for(int i_proc=0; i_proc<numprocs; ++i_proc){
            if(i_proc == myid){
                std::ostringstream oss;
                oss_green(oss, " -- check for output buffer");
                oss << " at proc=" << myid << "\n";
                printf(oss.str().c_str());

                std::vector<OpenFFT::dcomplex> buf, buf_ref;
                buf_ref.resize( fft_mngr.get_n_grid_out() );
                fft_mngr.apply_3d_array_with_output_buffer( &(Output[0][0][0]), buf_ref, OpenFFT::CopyIntoBuffer{} );

                print_green("    Manager<>::gen_3d_output_index_sequence()\n");

                fft_mngr.gen_3d_output_index_sequence(index_seq);
                buf.clear();
                for(const auto& index : index_seq){
                    buf.push_back( Output[ index[0] ][ index[1] ][ index[2] ] );
                }
                TEST::check_buffer(My_NumGrid_Out,
                                   buf.data(),
                                   buf_ref.data(), myid );

                for(int tgt_proc=0; tgt_proc<numprocs; ++tgt_proc){
                    const int n_grid_out = fft_mngr.get_n_grid_out(tgt_proc);
                    buf_ref.resize( n_grid_out );
                    fft_mngr.apply_3d_array_with_output_buffer( &(Output[0][0][0]), buf_ref, OpenFFT::CopyIntoBuffer{}, tgt_proc);

                    print_green("    Manager<>::gen_3d_output_index_sequence( tgt_proc )");
                    printf(", tgt_proc=%d\n", tgt_proc);

                    fft_mngr.gen_3d_output_index_sequence(index_seq, tgt_proc);
                    buf.clear();
                    for(const auto& index : index_seq){
                        buf.push_back( Output[ index[0] ][ index[1] ][ index[2] ] );
                    }
                    TEST::check_buffer(n_grid_out,
                                       buf.data(),
                                       buf_ref.data(), myid );
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    /* Finalize OpenFFT */

    fft_mngr.finalize();

    MPI_Finalize();

}
