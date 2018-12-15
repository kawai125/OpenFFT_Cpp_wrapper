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


template <class T>
void check_3d_array(const int n1, const int n2, const int n3,
                    const T *array_3d,
                    const T *ref_arr  ){
    bool check_flag = true;
    for(int iz=0; iz<n1; ++iz){
        for(int iy=0; iy<n2; ++iy){
            for(int ix=0; ix<n3; ++ix){
                const int pos = iz*(n2*n3) + iy*n3 + ix;

                const auto elem = array_3d[pos];
                const auto ref  = ref_arr[pos];
                if( std::abs(elem.r - ref.r) > 0.001 ||
                    std::abs(elem.i - ref.i) > 0.001   ){

                    printf("ERROR array[%d,%d,%d] data=(% 3.3f,% 3.3f), ref=(% 3.3f,% 3.3f)\n",
                    iz, iy, ix, elem.r, elem.i, ref.r, ref.i);
                    check_flag = false;
                }
            }
        }
    }
    if(check_flag){
        printf("   Check done. All elements are correct.\n");
    } else {
        printf("   Check done. Some elements are incorrect.\n");
    }
}
template <class T>
void check_buffer(const int  n_grid,
                  const T   *buf,
                  const T   *buf_ref,
                  const int  my_rank ){

    std::ostringstream oss;

    bool check_flag = true;
    for(int ii=0; ii<n_grid; ++ii){
        const auto elem = buf[ii];
        const auto ref  = buf_ref[ii];
        if( std::abs(elem.r - ref.r) > 0.001 ||
            std::abs(elem.i - ref.i) > 0.001   ){
            oss << "  ERROR in buffer[" << ii << "]: data=("
                << std::setprecision(3) << elem.r << ","
                << std::setprecision(3) << elem.i << "), ref=("
                << std::setprecision(3) << ref.r  << ","
                << std::setprecision(3) << ref.i  << ")\n";
            check_flag = false;
        }
    }
    if( ! check_flag ){
        oss << "   Check done. Some elements are incorrect at proc=" << my_rank << "\n";
    } else {
        oss << "   Check done. All elements are correct at proc=" << my_rank << "\n";
    }
    std::cout << oss.str() << std::flush;
}


int main(int argc, char* argv[])
{
    int numprocs,myid;
    int const N1=2,N2=3,N3=4;
    int offt_measure,measure_time,print_memory;
    int i,j,k;
    double factor;

    OpenFFT::dcomplex Input[N1][N2][N3],Output[N1][N2][N3];
    OpenFFT::dcomplex Out[N1][N2][N3]  ,Output_ref[N1][N2][N3];
    OpenFFT::dcomplex IFFT_Output[N1][N2][N3];

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

    //--- report internal table info (used for Manager<>::convert_output_to_input() function)
    // fft_mngr.report_convert_matrix();

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
    fft_mngr.copy_3d_array_into_input_buffer( &(Input[0][0][0]) , input_buffer);


    /* FFT transform */

    //------ call exec fft through OpenFFT::Manager
    //          "buffer" argument accepts std::vector<OpenFFT::dcomplex> or pointer <OpenFFT::dcomplex*>.
    fft_mngr.fft_c2c_3d_forward(input_buffer, output_buffer);


    /* Get local output */

    MPI_Barrier(MPI_COMM_WORLD);
    for(int i_proc=0; i_proc<numprocs; ++i_proc){
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
    fft_mngr.copy_3d_array_from_output_buffer( &(Out[0][0][0]) , output_buffer);

    for(i=0;i<N1;i++){
        for(j=0;j<N2;j++){
            for(k=0;k<N3;k++){
                Out[i][j][k].r /= factor;
                Out[i][j][k].i /= factor;
            }
        }
    }

    /* another copy method: using functor and "apply" API
    struct CopyFromBufferWithCalc{
        double n_inv_sqrt;
        void operator () (OpenFFT::dcomplex &arr_v, const OpenFFT::dcomplex &buf_v){
            arr_v.r = buf_v.r * this->n_inv_sqrt;
            arr_v.i = buf_v.i * this->n_inv_sqrt;
        }
    };
    CopyFromBufferWithCalc copy_from_buffer_with_calc;
    copy_from_buffer_with_calc.n_inv_sqrt = 1.0/factor;
    fft_mngr.apply_3d_array_with_output_buffer( &(Out[0][0][0]),
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


    /* Gather results from all processes */

    MPI_Allreduce(Out, Output, N1*N2*N3,
                  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    if(myid == 0){
        printf("\n");
        printf(" --- check FFT output ( using copy_3d_array_from_output_buffer() & MPI_Allreduce() )\n");
        check_3d_array(N1, N2, N3,
                       &(Output[0][0][0]), &(Output_ref[0][0][0]) );
    }

    fft_mngr.gather_3d_array( &(Output[0][0][0]), output_buffer, 0 );

    for(i=0;i<N1;i++){
        for(j=0;j<N2;j++){
            for(k=0;k<N3;k++){
                Output[i][j][k].r /= factor;
                Output[i][j][k].i /= factor;
            }
        }
    }

    if(myid == 0){
        printf("\n");
        printf(" --- check FFT output ( using Manager::gather_3d_array() )\n");
        check_3d_array(N1, N2, N3,
                       &(Output[0][0][0]), &(Output_ref[0][0][0]) );
    }

    fft_mngr.allgather_3d_array( &(Output[0][0][0]), output_buffer );

    for(i=0;i<N1;i++){
        for(j=0;j<N2;j++){
            for(k=0;k<N3;k++){
                Output[i][j][k].r /= factor;
                Output[i][j][k].i /= factor;
            }
        }
    }

    if(myid == 0){
        printf("\n");
        printf(" --- check FFT output ( using Manager::allgather_3d_array() )\n");
        check_3d_array(N1, N2, N3,
                       &(Output[0][0][0]), &(Output_ref[0][0][0]) );
    }

    //--- inverse FFT test
    MPI_Barrier(MPI_COMM_WORLD);
    if(myid == 0){
        printf("\n");
        printf("Inverse FFT test.\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //--- convert output buffer into input buffer.
    fft_mngr.convert_output_to_input(input_buffer, output_buffer);


    //--- check converted input_buffer
    std::vector<OpenFFT::dcomplex> ifft_input_buffer_ref;

    //------ this API automatically resize std::vector<>
    fft_mngr.copy_3d_array_into_input_buffer( &(Output[0][0][0]), ifft_input_buffer_ref );
    //------ cancel "factor"
    for(auto& elem : ifft_input_buffer_ref){
        elem.r *= factor;
        elem.i *= factor;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(myid == 0) printf("\n");
    for(int i_proc=0; i_proc<numprocs; ++i_proc){
        if(i_proc == myid){
            printf(" -- check converted Inverse FFT input buffer at proc=%d\n", myid);

            check_buffer(My_NumGrid_In,
                         input_buffer.data(),
                         ifft_input_buffer_ref.data(), myid );
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //--- perform Inverse FFT
    fft_mngr.fft_c2c_3d_backward(input_buffer, output_buffer);

    //--- collect result
    for(i=0;i<N1;i++){
        for(j=0;j<N2;j++){
            for(k=0;k<N3;k++){
                IFFT_Output[i][j][k].r = 0.0;
                IFFT_Output[i][j][k].i = 0.0;
            }
        }
    }
    fft_mngr.allgather_3d_array( &(IFFT_Output[0][0][0]) , output_buffer);

    //--- check Inverse FFT result
    MPI_Barrier(MPI_COMM_WORLD);
    if(myid == 0){
        printf("\n");
        printf(" -- check inverse FFT result at proc=%d\n", myid);

        check_3d_array(N1, N2, N3,
                       &(IFFT_Output[0][0][0]), &(Input[0][0][0]) );
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Finalize OpenFFT */
    fft_mngr.finalize();

    MPI_Finalize();

}
