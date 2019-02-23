/*******************************************************
*    This file is test for FFT c2c 3D interface
*       based on the OpenFFT sample code: check_c2c_3d.c
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
    int offt_measure,measure_time,print_memory;
    int i,j,k;
    double factor;

    OpenFFT::dcomplex Input[N1][N2][N3],Output[N1][N2][N3];
    OpenFFT::dcomplex Out[N1][N2][N3]  ,Output_ref[N1][N2][N3];

    /* MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    /* Set global input */

    Input[0][0][0].r = 1.000; Input[0][0][0].i = 0.000;
    Input[0][0][1].r = 0.999; Input[0][0][1].i =-0.040;
    Input[0][0][2].r = 0.987; Input[0][0][2].i =-0.159;
    Input[0][0][3].r = 0.936; Input[0][0][3].i =-0.352;
    Input[0][1][0].r = 0.994; Input[0][1][0].i =-0.111;
    Input[0][1][1].r = 0.989; Input[0][1][1].i =-0.151;
    Input[0][1][2].r = 0.963; Input[0][1][2].i =-0.268;
    Input[0][1][3].r = 0.891; Input[0][1][3].i =-0.454;
    Input[0][2][0].r = 0.903; Input[0][2][0].i =-0.430;
    Input[0][2][1].r = 0.885; Input[0][2][1].i =-0.466;
    Input[0][2][2].r = 0.823; Input[0][2][2].i =-0.568;
    Input[0][2][3].r = 0.694; Input[0][2][3].i =-0.720;
    Input[1][0][0].r = 0.500; Input[1][0][0].i = 0.500;
    Input[1][0][1].r = 0.499; Input[1][0][1].i = 0.040;
    Input[1][0][2].r = 0.487; Input[1][0][2].i = 0.159;
    Input[1][0][3].r = 0.436; Input[1][0][3].i = 0.352;
    Input[1][1][0].r = 0.494; Input[1][1][0].i = 0.111;
    Input[1][1][1].r = 0.489; Input[1][1][1].i = 0.151;
    Input[1][1][2].r = 0.463; Input[1][1][2].i = 0.268;
    Input[1][1][3].r = 0.391; Input[1][1][3].i = 0.454;
    Input[1][2][0].r = 0.403; Input[1][2][0].i = 0.430;
    Input[1][2][1].r = 0.385; Input[1][2][1].i = 0.466;
    Input[1][2][2].r = 0.323; Input[1][2][2].i = 0.568;
    Input[1][2][3].r = 0.194; Input[1][2][3].i = 0.720;

    /* Select auto-tuning of communication */

    offt_measure = 0;

    /* Set whether to use the timing and print memory functions of OpenFFT
       or not. Default=0 (not use) */

    measure_time = 0;
    print_memory = 0;

    /* Initialize OpenFFT */
    OpenFFT::Manager<double> fft_mngr;
    fft_mngr.init_c2c_3d(N1, N2, N3,
                         offt_measure, measure_time, print_memory);

    const auto My_Max_NumGrid = fft_mngr.get_max_n_grid();
    const auto My_NumGrid_In  = fft_mngr.get_n_grid_in();
    const auto My_Index_In    = fft_mngr.get_index_in();
    const auto My_NumGrid_Out = fft_mngr.get_n_grid_out();
    const auto My_Index_Out   = fft_mngr.get_index_out();

    /* Set local input */

    MPI_Barrier(MPI_COMM_WORLD);
    for(int i_proc=0; i_proc<numprocs; ++i_proc){
        if(i_proc == myid){
            printf("myid=%4d: Input in the ABC(XYZ) order with %d grid points ",
        	       myid,My_NumGrid_In);
            if(My_NumGrid_In > 0){
                printf("from (A=%d,B=%d,C=%d) to (A=%d,B=%d,C=%d)\n",
        	        My_Index_In[0],My_Index_In[1],My_Index_In[2],
        	        My_Index_In[3],My_Index_In[4],My_Index_In[5]);
            } else {
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Print global input */

    if(!myid){
        printf("Input values\n\n");
        for(i=0;i<N1;i++){
            printf("Input(i,j,k) for i = %d\n\n",i);
            for(j=0;j<N2;j++){
                printf("Real\t");
                for(k=0;k<N3;k++){
                    printf("% 3.3f\t",Input[i][j][k].r);
                }
                printf("\nImag\t");
                for(k=0;k<N3;k++){
                    printf("% 3.3f\t",Input[i][j][k].i);
                }
                printf("\n\n");
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);


    //--- perform FFT through OpenFFT::Manager<>
    std::vector<OpenFFT::dcomplex> input_buffer, output_buffer;
    input_buffer.reserve( My_Max_NumGrid);
    output_buffer.reserve(My_Max_NumGrid);
    input_buffer.resize( My_NumGrid_In );
    output_buffer.resize(My_NumGrid_Out);
    std::fill( input_buffer.begin() , input_buffer.end() , OpenFFT::dcomplex{0.0, 0.0} );
    std::fill( output_buffer.begin(), output_buffer.end(), OpenFFT::dcomplex{0.0, 0.0} );

    //------ copy 3D array data into local input buffer
    fft_mngr.copy_array_into_input_buffer( &(Input[0][0][0]) , input_buffer);


    /* FFT transform */

    //------ call exec fft through OpenFFT::Manager
    //          "buffer" argument accepts std::vector<OpenFFT::dcomplex> or pointer <OpenFFT::dcomplex*>.
    fft_mngr.fft_c2c_forward(input_buffer, output_buffer);


    /* Get local output */

    for(int i_proc=0; i_proc<numprocs; ++i_proc){
        MPI_Barrier(MPI_COMM_WORLD);
        if(i_proc == myid){
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
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    factor = sqrt(N1*N2*N3);

    for(i=0;i<N1;i++){
        for(j=0;j<N2;j++){
            for(k=0;k<N3;k++){
                Out[i][j][k].r = 0.0;
                Out[i][j][k].i = 0.0;
                Output[i][j][k].r = 0.0;
                Output[i][j][k].i = 0.0;
            }
        }
    }

    //------ local output buffer into 3D-array
    fft_mngr.copy_array_from_output_buffer( &(Out[0][0][0]) , output_buffer);

    for(i=0;i<N1;i++){
        for(j=0;j<N2;j++){
            for(k=0;k<N3;k++){
                Out[i][j][k].r /= factor;
                Out[i][j][k].i /= factor;
            }
        }
    }

    /* another copy method: using functor with "apply" API
    struct CopyFromBufferWithCalc{
        double n_inv_sqrt;
        void operator () (OpenFFT::dcomplex &arr_v, const OpenFFT::dcomplex &buf_v){
            arr_v.r = buf_v.r * this->n_inv_sqrt;
            arr_v.i = buf_v.i * this->n_inv_sqrt;
        }
    };
    CopyFromBufferWithCalc copy_from_buffer_with_calc;
    copy_from_buffer_with_calc.n_inv_sqrt = 1.0/factor;
    fft_mngr.apply_array_with_output_buffer( &(Out[0][0][0]),
                                               output_buffer,
                                               copy_from_buffer_with_calc);
    */

    /* Print global output */

    if(!myid){
        printf("Output values\n\n");
        for(i=0;i<N1;i++){
            printf("Output(i,j,k) for i = %d\n\n",i);
            for(j=0;j<N2;j++){
                printf("Real\t");
                for(k=0;k<N3;k++){
                    printf("% 3.3f\t",Output[i][j][k].r);
                }
                printf("\nImag\t");
                for(k=0;k<N3;k++){
                    printf("% 3.3f\t",Output[i][j][k].i);
                }
                printf("\n\n");
            }
        }
    }

    /* Error check */

    Output_ref[0][0][0].r = 3.292; Output_ref[0][0][0].i = 0.102;
    Output_ref[0][0][1].r = 0.051; Output_ref[0][0][1].i =-0.042;
    Output_ref[0][0][2].r = 0.113; Output_ref[0][0][2].i = 0.102;
    Output_ref[0][0][3].r = 0.051; Output_ref[0][0][3].i = 0.246;
    Output_ref[0][1][0].r = 0.143; Output_ref[0][1][0].i =-0.086;
    Output_ref[0][1][1].r = 0.016; Output_ref[0][1][1].i = 0.153;
    Output_ref[0][1][2].r =-0.024; Output_ref[0][1][2].i = 0.127;
    Output_ref[0][1][3].r =-0.050; Output_ref[0][1][3].i = 0.086;
    Output_ref[0][2][0].r = 0.143; Output_ref[0][2][0].i = 0.290;
    Output_ref[0][2][1].r =-0.050; Output_ref[0][2][1].i = 0.118;
    Output_ref[0][2][2].r =-0.024; Output_ref[0][2][2].i = 0.077;
    Output_ref[0][2][3].r = 0.016; Output_ref[0][2][3].i = 0.051;
    Output_ref[1][0][0].r = 1.225; Output_ref[1][0][0].i =-1.620;
    Output_ref[1][0][1].r = 0.355; Output_ref[1][0][1].i = 0.083;
    Output_ref[1][0][2].r = 0.000; Output_ref[1][0][2].i = 0.162;
    Output_ref[1][0][3].r =-0.355; Output_ref[1][0][3].i = 0.083;
    Output_ref[1][1][0].r = 0.424; Output_ref[1][1][0].i = 0.320;
    Output_ref[1][1][1].r = 0.020; Output_ref[1][1][1].i =-0.115;
    Output_ref[1][1][2].r = 0.013; Output_ref[1][1][2].i =-0.091;
    Output_ref[1][1][3].r =-0.007; Output_ref[1][1][3].i =-0.080;
    Output_ref[1][2][0].r =-0.424; Output_ref[1][2][0].i = 0.320;
    Output_ref[1][2][1].r = 0.007; Output_ref[1][2][1].i =-0.080;
    Output_ref[1][2][2].r =-0.013; Output_ref[1][2][2].i =-0.091;
    Output_ref[1][2][3].r =-0.020; Output_ref[1][2][3].i =-0.115;

    //==========================================================================
    //  Gather results interface test
    //==========================================================================
    {
        MPI_Allreduce(Out, Output, N1*N2*N3,
                      MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            printf(     "\n");
            print_green("[checking FFT output]\n");
            printf(     "\n");
            print_green("   Manager<>::copy_array_from_output_buffer() & MPI_Allreduce()\n");
            TEST::check_3d_array(N1, N2, N3,
                                 &(Output[0][0][0]), &(Output_ref[0][0][0]) );
            printf(     "\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        for(int i_proc=0; i_proc<numprocs; ++i_proc){
            fft_mngr.gather_array( &(Output[0][0][0]), output_buffer, i_proc );

            MPI_Barrier(MPI_COMM_WORLD);
            if(myid == i_proc){
                print_green("   Manager<>::gather_array()");
                printf(     " at proc=%d\n", i_proc);

                for(i=0;i<N1;i++){
                    for(j=0;j<N2;j++){
                        for(k=0;k<N3;k++){
                            Output[i][j][k].r /= factor;
                            Output[i][j][k].i /= factor;
                        }
                    }
                }

                TEST::check_3d_array(N1, N2, N3,
                                     &(Output[0][0][0]), &(Output_ref[0][0][0]) );
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }


        if(myid == 0){
            printf(     "\n");
            print_green("   Manager<>::allgather_array() )\n");
        }

        fft_mngr.allgather_array( &(Output[0][0][0]), output_buffer );

        for(i=0;i<N1;i++){
            for(j=0;j<N2;j++){
                for(k=0;k<N3;k++){
                    Output[i][j][k].r /= factor;
                    Output[i][j][k].i /= factor;
                }
            }
        }

        if(myid == 0){
            TEST::check_3d_array(N1, N2, N3,
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

        OpenFFT::dcomplex sum_v;
        OpenFFT::dcomplex sum_v_ref;

        sum_v_ref.r = 0.0;
        sum_v_ref.i = 0.0;
        for(i=0;i<N1;i++){
            for(j=0;j<N2;j++){
                for(k=0;k<N3;k++){
                     sum_v_ref.r += Input[i][j][k].r * Output[i][j][k].r;
                     sum_v_ref.i += Input[i][j][k].i * Output[i][j][k].i;
                }
            }
        }

        struct ReduceValue{
            OpenFFT::dcomplex v {0.0, 0.0};
            double            factor = 1.0;
            void operator () (const OpenFFT::dcomplex arr_v, const OpenFFT::dcomplex buf_v){
                this->v.r += arr_v.r * buf_v.r / this->factor;
                this->v.i += arr_v.i * buf_v.i / this->factor;
            }
        };
        ReduceValue reduce_value;

        reduce_value.factor = 1.0;
        fft_mngr.copy_array_into_input_buffer( &(Input[0][0][0]), input_buffer);
        const auto local_sum_ib = fft_mngr.apply_array_with_input_buffer( &(Output[0][0][0]),
                                                                            input_buffer,
                                                                            reduce_value );

        MPI_Allreduce(&local_sum_ib.v, &sum_v, 1,
                      MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

        if(myid == 0){
            bool check_flag = true;
            if( std::abs(sum_v.r - sum_v_ref.r) > 0.0001 ||
                std::abs(sum_v.i - sum_v_ref.i) > 0.0001   ){
                check_flag = false;
                print_yellow("  ERROR:");
                printf(" reduced sum_v = (% 6.5f,% 6.5f), ref = (% 6.5f,% 6.5f)\n",
                       sum_v.r, sum_v.i, sum_v_ref.r, sum_v_ref.i);
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

        reduce_value.v.r    = 0.0;
        reduce_value.v.i    = 0.0;
        reduce_value.factor = factor;
        const auto local_sum_ob = fft_mngr.apply_array_with_output_buffer( &(Input[0][0][0]),
                                                                             output_buffer,
                                                                             reduce_value );

        MPI_Allreduce(&local_sum_ob.v, &sum_v, 1,
                      MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

        if(myid == 0){
            bool check_flag = true;
            if( std::abs(sum_v.r - sum_v_ref.r) > 0.0001 ||
                std::abs(sum_v.i - sum_v_ref.i) > 0.0001   ){
                check_flag = false;
                print_yellow("  ERROR:");
                printf(" reduced sum_v = (% 6.5f,% 6.5f), ref = (% 6.5f,% 6.5f)\n",
                       sum_v.r, sum_v.i, sum_v_ref.r, sum_v_ref.i);
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
    //  transpose function test
    //==========================================================================
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            printf("\n");
            print_green("[transpose function test]\n");
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        {
            std::vector<OpenFFT::dcomplex> input_buf;
            std::vector<OpenFFT::dcomplex> in_out_convert;
            std::vector<OpenFFT::dcomplex> in_out_convert_ref;

            in_out_convert_ref.resize( fft_mngr.get_n_grid_out() );
            fft_mngr.copy_array_into_output_buffer( &(Input[0][0][0]), in_out_convert_ref );

            fft_mngr.copy_array_into_input_buffer( &(Input[0][0][0]), input_buf );

            fft_mngr.transpose_input_to_output(input_buf, in_out_convert);

            MPI_Barrier(MPI_COMM_WORLD);
            for(int i_proc=0; i_proc<numprocs; ++i_proc){
                MPI_Barrier(MPI_COMM_WORLD);
                if(i_proc == myid){
                    std::ostringstream oss;
                    oss_green(oss, "   Manager<>::transpose_input_to_output()");
                    oss << " at proc=" << myid << "\n";
                    printf(oss.str().c_str());

                    TEST::check_buffer(fft_mngr.get_n_grid_out(),
                                       in_out_convert.data(),
                                       in_out_convert_ref.data(), myid );
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        if(myid == 0) printf("\n");
        {
            std::vector<OpenFFT::dcomplex> output_buf;
            std::vector<OpenFFT::dcomplex> out_in_convert;
            std::vector<OpenFFT::dcomplex> out_in_convert_ref;

            fft_mngr.copy_array_into_input_buffer( &(Output[0][0][0]), out_in_convert_ref );

            output_buf.resize( fft_mngr.get_n_grid_out() );
            fft_mngr.copy_array_into_output_buffer( &(Output[0][0][0]), output_buf );

            fft_mngr.transpose_output_to_input(output_buf, out_in_convert);

            MPI_Barrier(MPI_COMM_WORLD);
            for(int i_proc=0; i_proc<numprocs; ++i_proc){
                MPI_Barrier(MPI_COMM_WORLD);
                if(i_proc == myid){
                    std::ostringstream oss;
                    oss_green(oss, "   Manager<>::convert_output_to_input()");
                    oss << " at proc=" << myid << "\n";
                    printf(oss.str().c_str());

                    TEST::check_buffer(fft_mngr.get_n_grid_in(),
                                       out_in_convert.data(),
                                       out_in_convert_ref.data(), myid );
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
    }

    //==========================================================================
    //  inverse FFT test
    //==========================================================================
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            printf("\n");
            print_green("[inverse FFT test]\n");
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        OpenFFT::dcomplex IFFT_Output[N1][N2][N3];

        //--- convert output buffer into input buffer.
        fft_mngr.transpose_output_to_input(output_buffer, input_buffer);

        //--- perform Inverse FFT
        fft_mngr.fft_c2c_backward(input_buffer, output_buffer);

        //--- collect result
        fft_mngr.allgather_array( &(IFFT_Output[0][0][0]) , output_buffer);

        //--- normalize Inverse FFT result
        const double N_inv = 1.0/static_cast<double>(N1*N2*N3);
        for(i=0;i<N1;i++){
            for(j=0;j<N2;j++){
                for(k=0;k<N3;k++){
                    IFFT_Output[i][j][k].r *= N_inv;
                    IFFT_Output[i][j][k].i *= N_inv;
                }
            }
        }

        //--- check Inverse FFT result
        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            std::ostringstream oss;
            oss_green(oss, "   Manager<>::transpose_output_to_input() & Manager<>::fft_c2c_backward()");
            oss << " at proc=" << myid << "\n";
            printf(oss.str().c_str());

            TEST::check_3d_array(N1, N2, N3,
                                 &(IFFT_Output[0][0][0]), &(Input[0][0][0]) );
        }
        MPI_Barrier(MPI_COMM_WORLD);
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
            MPI_Barrier(MPI_COMM_WORLD);
            if(i_proc == myid){
                std::ostringstream oss;
                oss_green(oss, " -- check index for input buffer");
                oss << " at proc=" << myid << "\n";
                printf(oss.str().c_str());

                std::vector<OpenFFT::dcomplex> buf, buf_ref;
                fft_mngr.copy_array_into_input_buffer( &(Input[0][0][0]), buf_ref);

                print_green("    Manager<>::gen_input_index_sequence()\n");
                fft_mngr.gen_input_index_sequence(index_seq);
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
                    fft_mngr.apply_array_with_input_buffer( &(Input[0][0][0]), buf_ref, OpenFFT::CopyIntoBuffer{}, tgt_proc);

                    print_green("    Manager<>::gen_input_index_sequence( tgt_proc )");
                    printf(", tgt_proc=%d\n", tgt_proc);

                    fft_mngr.gen_input_index_sequence(index_seq, tgt_proc);
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
            MPI_Barrier(MPI_COMM_WORLD);
            if(i_proc == myid){
                std::ostringstream oss;
                oss_green(oss, " -- check for output buffer");
                oss << " at proc=" << myid << "\n";
                printf(oss.str().c_str());

                std::vector<OpenFFT::dcomplex> buf, buf_ref;
                buf_ref.resize( fft_mngr.get_n_grid_out() );
                fft_mngr.copy_array_into_output_buffer( &(Output[0][0][0]), buf_ref );

                print_green("    Manager<>::gen_output_index_sequence()\n");

                fft_mngr.gen_output_index_sequence(index_seq);
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
                    fft_mngr.apply_array_with_output_buffer( &(Output[0][0][0]), buf_ref, OpenFFT::CopyIntoBuffer{}, tgt_proc);

                    print_green("    Manager<>::gen_output_index_sequence( tgt_proc )");
                    printf(", tgt_proc=%d\n", tgt_proc);

                    fft_mngr.gen_output_index_sequence(index_seq, tgt_proc);
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

    //--- report test result
    const int final_state = TEST::report();

    MPI_Finalize();

    return final_state;
}
