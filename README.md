# OpenFFT_Cpp_wrapper

## Introduction
C++ interface & utilities for [OpenFFT](http://www.openmx-square.org/openfft/) library.  
This wrapper provides array-index management between global 3D/4D array data and distributed buffer data for OpenFFT.

## License
The implementations of OpenFFT_Cpp_wrapper (inside `./include` directory) are released under the MIT license.  
The sample codes (inside `./sample` directory) are released under the GPLv3 or any later version license because they written based on the original sample codes of the OpenFFT library.

## How to use
Include the `./include/openfft.hpp` in your source code, then compile and link with the `libopenfft.a` of OpenFFT.  
__This library requires compatibility of C++11 standard for C++ compiler.__

## Calling wrapper function from your C++ program
 - Definition of Interface.  
   All definitions are implemented in the namespace of `OpenFFT::` .  

   The complex type for OpenFFT library is `OpenFFT::dcomplex` .  The members of dcomplex are shown in below.  
   ```c++
   OpenFFT::dcomplex complex_data;

   double real_part = complex_data.r;
   double imag_part = complex_data.i;
   ```

   The management class for OpenFFT library is `OpenFFT::Manager<double>` .  
   In current version of OpenFFT, that is only implemented in 64bit float (double) version.

   Data type and index type informations are available from manager class.
   ```c++
   using float_type   = typename OpenFFT::Manager<double>::float_type;    //  same to double
   using complex_type = typename OpenFFT::Manager<double>::complex_type;  //  same to OpenFFT::dcomplex
   using index_type   = typename OpenFFT::Manager<double>::IndexList;     //  same to std::array<int, 8>
   ```

 - Step 1: Initialize manager.  
   ```c++
   //--- make manager instance
   OpenFFT::Manager<double> fft_mngr;

   //--- initialize for r2c 3D FFT
   fft_mngr.init_r2c_3d(N1, N2, N3,
                        offt_measure, measure_time, print_memory);

   //--- initialize for c2c 3D FFT
   fft_mngr.init_c2c_3d(N1, N2, N3,
                        offt_measure, measure_time, print_memory);

   //--- initialize for c2c 4D FFT
   fft_mngr.init_c2c_4d(N1, N2, N3, N4,
                        offt_measure, measure_time, print_memory);
   ```
   These grid size and transform type are shared in global manager (that is static instance in this wrapper). You can select each one configuration exclusively in same time. (re-initialize to different configuration is possible.)

  - Step 2: Copy the global 3D/4D array into local input buffer.  
    ```c++
    //--- copy data for r2c 3D FFT
    double RealGlobalInput[N1][N2][N3];
    std::vector<double> real_input_buffer;
    fft_mngr.copy_3d_array_into_input_buffer( &( RealGlobalInput[0][0][0] ), real_input_buffer );

    //--- copy data for c2c 3D FFT
    OpenFFT::dcomplex GlobalInput[N1][N2][N3];
    std::vector<OpenFFT::dcomplex> input_buffer;
    fft_mngr.copy_3d_array_into_input_buffer( &( GlobalInput[0][0][0] ), input_buffer );

    //--- copy data for c2c 4D FFT
    OpenFFT::dcomplex GlobalInput[N1][N2][N3][N4];
    std::vector<OpenFFT::dcomplex> input_buffer;
    fft_mngr.copy_4d_array_into_input_buffer( &( GlobalInput[0][0][0][0] ), input_buffer );
    ```
    All 'buffer' input/output accepts both of reference to `std::vector<T>` and pointer to buffer array (`double *` or `OpenFFT::dcomplex *`).  
    Using `std::vector<T>` is recommended because the buffer length is checked for input buffer and resized for output buffer in this API.

  - Step 3: Execute FFT.  
    ```c++
    //--- for r2c 3D FFT
    std::vector<OpenFFT::dcomplex> output_buffer;
    fft_mngr.fft_r2c_3d_forward( real_input_buffer, output_buffer );

    //--- for c2c 3D FFT
    std::vector<OpenFFT::dcomplex> output_buffer;
    fft_mngr.fft_c2c_3d_forward( input_buffer, output_buffer );

    //--- for c2c 4D FFT
    std::vector<OpenFFT::dcomplex> output_buffer;
    fft_mngr.fft_c2c_4d_forward( input_buffer, output_buffer );
    ```

  - Step 4: write back local output buffer data into the local 3D/4D array.  
    ```c++
    //--- write back for r2c 3D FFT
    OpenFFT::dcomplex LocalOutput[N1][N2][N3r];   // N3r = N3/2+1
    fft_mngr.copy_3d_array_from_output_buffer( &( GlobalOutput[0][0][0] ), output_buffer);

    //--- write back for c2c 3D FFT
    OpenFFT::dcomplex LocalOutput[N1][N2][N3];
    fft_mngr.copy_3d_array_from_output_buffer( &( GlobalOutput[0][0][0] ), output_buffer);

    //--- write back for c2c 4D FFT
    OpenFFT::dcomplex LocalOutput[N1][N2][N3][N4];
    fft_mngr.copy_4d_array_from_output_buffer( &( GlobalOutput[0][0][0][0] ), output_buffer);
    ```

   - Step 4-2: Convert output buffer into input buffer for Inverse FFT (available in c2c_3D mode only).  
     ```c++
     fft_mngr.convert_output_to_input( input_buffer, output_buffer );
     ```
     It communicate 'output_buffer' by using MPI_Alltoallv() and shape received data into input_buffer.

   - Step 4-3: execute Inverse FFT (available in c2c_3D mode only).  
     ```c++
     fft_mngr.fft_c2c_3d_backward( input_buffer, output_buffer );
     ```
     This function is compatible with backward DFT compute of FFTW3.

   - step 5: Finalize.  
     ```c++
     fft_mngr.finalize();
     ```
     The OpenFFT configurations are shared in global manager object (It allows you to make many instance of OpenFFT::Manager<>).  
     It is recommended that you call the 'finalize' function at once in finalization part of your program.

## Other APIs

   - Get OpenFFT information of local process.  
     ```c++
     int                My_Max_NumGrid = fft_mngr.get_max_n_grid();
     int                My_NumGrid_In  = fft_mngr.get_n_grid_in();
     int                My_NumGrid_Out = fft_mngr.get_n_grid_out();
     std::array<int, 8> My_Index_In    = fft_mngr.get_index_in();
     std::array<int, 8> My_Index_Out   = fft_mngr.get_index_out();
     ```

   - Get OpenFFT information of other process.  
     ```c++
     int tgt_proc;  // your target MPI process ID

     int                NumGrid_In  = fft_mngr.get_n_grid_in(  tgt_proc );
     int                NumGrid_Out = fft_mngr.get_n_grid_out( tgt_proc );
     std::array<int, 8> Index_In    = fft_mngr.get_index_in(   tgt_proc );
     std::array<int, 8> Index_Out   = fft_mngr.get_index_out(  tgt_proc );
     ```

   - Gather functions are available to build global 3D/4D array from local output buffer.  
     ```c++
     //--- gather for r2c 3D FFT and c2c 3D FFT
     OpenFFT::dcomplex GlobalOutput[N1][N2][N3];
     const int tgt_proc = 0; // MPI proc id
     fft_mngr.gather_3d_array( &( GlobalOutput[0][0][0] ), output_buffer, tgt_proc );

     //--- gather for c2c 4D FFT
     OpenFFT::dcomplex GlobalOutput[N1][N2][N3][N4];
     const int tgt_proc = 0; // MPI proc id
     fft_mngr.gather_4d_array( &( GlobalOutput[0][0][0][0] ), output_buffer, tgt_proc );


     //--- allgather for r2c 3D FFT and c2c 3D FFT
     OpenFFT::dcomplex GlobalOutput[N1][N2][N3];
     fft_mngr.allgather_3d_array( &( GlobalOutput[0][0][0] ), output_buffer );

     //--- allgather for c2c 4D FFT
     OpenFFT::dcomplex GlobalOutput[N1][N2][N3][N4];
     fft_mngr.allgather_4d_array( &( GlobalOutput[0][0][0][0] ), output_buffer );
     ```

   - Apply your function between global 3D/4D array and local input/output buffer.  
     ```c++
     //--- example 1: collect real part of output.
     double RealGlobalOutput[N1][N2][N3];

     //------ using functor
     struct CopyRealpartFromBuffer {
         void operator () (double &arr_v, const OpenFFT::dcomplex &buf_v){
             arr_v = buf_v.r;
         }
     };
     CopyRealpartFromBuffer your_func;

     fft_mngr.apply_3d_array_with_output_buffer( &(RealGlobalOutput[0][0][0]), output_buffer, your_func );



     //--- example 2: if you have the local buffer in other process.
     int proc_i;  // MPI process ID which has the part of 'output_buffer_at_proc_i'.
     std::vector<OpenFFT::dcomplex> output_buffer_at_proc_i;

     //------- this API accepts explicit MPI process ID.
     fft_mngr.apply_3d_array_with_output_buffer( &(RealGlobalOutput[0][0][0]), output_buffer_at_proc_i, your_func, proc_i);
     ```
     The implementations of `OpenFFT::Manager<>::apply_[3d/4d]_array_with_[input/output]_buffer()` are used in other API between global 3D/4D array and local buffer. These 'apply' functions can accept explicit MPI process ID.

     These 'apply' functions return the functor that you passed. It can use for reducing value from local buffer.  
     ```c++
     //--- calculate sum of element-wise product between ApplyMatrix array and Global output array.
     double ApplyMatrix[N1][N2][N3]
     std::vector<OpenFFT::dcomplex> output_buffer;

     //--- reduce real part product
     struct ReduceValue{
         double v = 0.0;
         void operator () (const double arr_v, const OpenFFT::dcomplex buf_v){
             this->v += arr_v*buf_v.r;
         }
     };
     ReduceValue reduce_value;

     //--- get local sum
     const ReduceValue local_sum = fft_mngr.apply_3d_array_with_output_buffer( &(ApplyMatrix[0][0][0]), output_buffer, reduce_value );

     //--- get global sum
     double global_sum;
     MPI_Allreduce( &local_sum.v, &global_sum, 1
                   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     ```

   - Display parameters for `convert_output_to_input()` (available in c2c_3D mode only).  
     ```c++
     fft_mngr.report_convert_matrix();
     ```
