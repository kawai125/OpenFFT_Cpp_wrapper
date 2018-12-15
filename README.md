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
   OpwnFFT::dcomplex complex_data;

   double real_part = complex_data.r;
   double imag_part = complex_data.i;
   ```

   The management class for OpenFFT library is `OpenFFT::Manager<double>` .  
   In current version of OpenFFT, 64bit float (double) version is only implemented.

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
    fft_mngr.copy_3d_array_into_input_buffer( &( GlobalInput[0][0][0][0] ), input_buffer );
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

  - Step 4: Gather local output buffer data into the global 3D/4D array.  
    ```c++
    //--- gather for r2c 3D FFT and c2c 3D FFT
    OpenFFT::dcomplex GlobalOutput[N1][N2][N3];
    const int tgt_proc = 0; // MPI proc id
    fft_mngr.gather_3d_array( &( GlobalOutput[0][0][0] ), output_buffer, tgt_proc );

    //--- gather c2c 4D FFT
    OpenFFT::dcomplex GlobalOutput[N1][N2][N3][N4];
    const int tgt_proc = 0; // MPI proc id
    fft_mngr.gather_4d_array( &( GlobalOutput[0][0][0][0] ), output_buffer, tgt_proc );
    ```
    Also `allgather_3d_array( global_output, output_buffer )`  
    and `allgather_4d_array( global_output, output_buffer )` functions are available to gather and broadcast 'GlobalOutput' data for all MPI processes.

   - Step 4-2: Convert output buffer into input buffer for Inverse FFT (available in c2c_3D mode only).  
     ```c++
     fft_mngr.convert_output_to_input( input_buffer, output_buffer );
     ```
     It communicate 'output_buffer' by using MPI_Alltoallv() and shape received data into input_buffer.

   - Step 4-3: execute Inverse FFT (available in c2c_3D mode only).  
     ```c++
     fft_mngr.fft_c2c_3d_backward( input_buffer, output_buffer );
     ```

   - step 5: Finalize.  
     ```c++
     fft_mngr.finalize();
     ```
     The OpenFFT configurations are shared in global manager object (It allows you to make many instance of OpenFFT::Manager<>). It is recommended that you call the 'finalize' function at once in 'finalize' part of your program.

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

   - Apply your function between Global 3D/4D array and local input/output buffer.  
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

   - Display parameters for `convert_output_to_input()` (available in c2c_3D mode only).  
     ```c++
     fft_mngr.report_convert_matrix();
     ```
