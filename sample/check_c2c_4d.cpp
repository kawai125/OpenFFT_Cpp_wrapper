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
    fft_mngr.copy_array_into_input_buffer( &(Input[0][0][0][0]) , input_buffer);


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

    fft_mngr.fft_c2c_forward( input_buffer, output_buffer );

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
    fft_mngr.copy_array_from_output_buffer( &(Out[0][0][0][0]) , output_buffer);

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
            print_green("    Manager<>::copy_array_from_output_buffer() & MPI_Allreduce()\n");
            TEST::check_4d_array(N1, N2, N3, N4,
                                 &(Output[0][0][0][0]), &(Output_ref[0][0][0][0]) );
        }

        if(myid == 0) printf(     "\n");
        for(int i_proc=0; i_proc<numprocs; ++i_proc){

            fft_mngr.gather_array( &(Output[0][0][0][0]), output_buffer, i_proc );

            MPI_Barrier(MPI_COMM_WORLD);
            if(myid == i_proc){
                print_green("    Manager<>::gather_array()");
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


        fft_mngr.allgather_array( &(Output[0][0][0][0]), output_buffer );

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
                print_green("    Manager<>::allgather_array()");
                printf(     " at proc=%d\n", i_proc);
                TEST::check_4d_array(N1, N2, N3, N4,
                                     &(Output[0][0][0][0]), &(Output_ref[0][0][0][0]) );
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
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
        for(m=0;m<N1;m++){
            for(i=0;i<N2;i++){
                for(j=0;j<N3;j++){
                    for(k=0;k<N4;k++){
                        sum_v_ref.r += Input[m][i][j][k].r * Output[m][i][j][k].r;
                        sum_v_ref.i += Input[m][i][j][k].i * Output[m][i][j][k].i;
                    }
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
        fft_mngr.copy_array_into_input_buffer( &(Input[0][0][0][0]), input_buffer);
        const auto local_sum_ib = fft_mngr.apply_array_with_input_buffer( &(Output[0][0][0][0]),
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
        const auto local_sum_ob = fft_mngr.apply_array_with_output_buffer( &(Input[0][0][0][0]),
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
            fft_mngr.copy_array_into_output_buffer( &(Input[0][0][0][0]), in_out_convert_ref );

            fft_mngr.copy_array_into_input_buffer( &(Input[0][0][0][0]), input_buf );

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

            fft_mngr.copy_array_into_input_buffer( &(Output[0][0][0][0]), out_in_convert_ref );

            output_buf.resize( fft_mngr.get_n_grid_out() );
            fft_mngr.copy_array_into_output_buffer( &(Output[0][0][0][0]), output_buf );

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

        OpenFFT::dcomplex IFFT_Output[N1][N2][N3][N4];

        //--- convert output buffer into input buffer.
        fft_mngr.transpose_output_to_input(output_buffer, input_buffer);

        //--- perform Inverse FFT
        fft_mngr.fft_c2c_backward(input_buffer, output_buffer);

        //--- collect result
        fft_mngr.allgather_array( &(IFFT_Output[0][0][0][0]) , output_buffer);

        //--- normalize Inverse FFT result
        const double N_inv = 1.0/static_cast<double>(N1*N2*N3*N4);
        for(i=0;i<N1;i++){
            for(j=0;j<N2;j++){
                for(k=0;k<N3;k++){
                    for(m=0;m<N4;m++){
                        IFFT_Output[i][j][k][m].r *= N_inv;
                        IFFT_Output[i][j][k][m].i *= N_inv;
                    }
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

            TEST::check_4d_array(N1, N2, N3, N4,
                                 &(IFFT_Output[0][0][0][0]), &(Input[0][0][0][0]) );
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //==========================================================================
    //  index sequence generator test
    //==========================================================================
    {
        std::vector<std::array<int, 4>> index_seq;

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

                std::vector<OpenFFT::dcomplex> buf, buf_ref;
                fft_mngr.copy_array_into_input_buffer( &(Input[0][0][0][0]), buf_ref);

                print_green("    Manager<>::gen_input_index_sequence()\n");
                fft_mngr.gen_input_index_sequence(index_seq);
                buf.clear();
                for(const auto& index : index_seq){
                    buf.push_back( Input[ index[0] ][ index[1] ][ index[2] ][ index[3] ] );
                }
                TEST::check_buffer(My_NumGrid_In,
                                   buf.data(),
                                   buf_ref.data(), myid );

                for(int tgt_proc=0; tgt_proc<numprocs; ++tgt_proc){
                    const int n_grid_in = fft_mngr.get_n_grid_in(tgt_proc);
                    buf_ref.resize( n_grid_in );
                    fft_mngr.apply_array_with_input_buffer( &(Input[0][0][0][0]), buf_ref, OpenFFT::CopyIntoBuffer{}, tgt_proc);

                    print_green("    Manager<>::gen_input_index_sequence( tgt_proc )");
                    printf(", tgt_proc=%d\n", tgt_proc);

                    fft_mngr.gen_input_index_sequence(index_seq, tgt_proc);
                    buf.clear();
                    for(const auto& index : index_seq){
                        buf.push_back( Input[ index[0] ][ index[1] ][ index[2] ][ index[3] ] );
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
                fft_mngr.copy_array_into_output_buffer( &(Output[0][0][0][0]), buf_ref );

                print_green("    Manager<>::gen_output_index_sequence()\n");

                fft_mngr.gen_output_index_sequence(index_seq);
                buf.clear();
                for(const auto& index : index_seq){
                    buf.push_back( Output[ index[0] ][ index[1] ][ index[2] ][ index[3] ] );
                }
                TEST::check_buffer(My_NumGrid_Out,
                                   buf.data(),
                                   buf_ref.data(), myid );

                for(int tgt_proc=0; tgt_proc<numprocs; ++tgt_proc){
                    const int n_grid_out = fft_mngr.get_n_grid_out(tgt_proc);
                    buf_ref.resize( n_grid_out );
                    fft_mngr.apply_array_with_output_buffer( &(Output[0][0][0][0]), buf_ref, OpenFFT::CopyIntoBuffer{}, tgt_proc);

                    print_green("    Manager<>::gen_output_index_sequence( tgt_proc )");
                    printf(", tgt_proc=%d\n", tgt_proc);

                    fft_mngr.gen_output_index_sequence(index_seq, tgt_proc);
                    buf.clear();
                    for(const auto& index : index_seq){
                        buf.push_back( Output[ index[0] ][ index[1] ][ index[2] ][ index[3] ] );
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
