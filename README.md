# OpenFFT_Cpp_wrapper

## Introduction
C++ interface & utilities for [OpenFFT](http://www.openmx-square.org/openfft/) library.  
This wrapper provides array-index management between global 3D/4D array data and distributed buffer data for OpenFFT.

## License
The implementations of OpenFFT_Cpp_wrapper (inside `./include` directory) are released under the MIT license.  
The sample codes (inside `./sample` directory) are released under the GPLv3 or any later version license because they written based on the original sample codes of the OpenFFT library.

## How to use
Include the `./include/openfft.hpp` in your source code, then compile and link with the `libopenfft.a` of OpenFFT.  
__Do not include__ the `openfft.h` of OpenFFT.

__This library uses C++11 standard.__

This library is developed in the environment shown in below.
 - GCC 6.4
 - OpenMPI 2.1.2
 - FFTW 3.3.7
 - OpenFFT 1.2

## Calling wrapper functions from your C++ program
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


   //--- initialize for c2c_3D FFT with default settings
   //      auto-tuning, without measure-time and print-memory.
   fft_mngr.init_c2c_3d(N1, N2, N3);
   ```
   These grid size and transform type are shared in global manager (that is static instance in this wrapper).  
   You can select each one configuration exclusively in same time. (re-initialize to different configuration is possible.)

  - Step 2: Copy the global 3D/4D array into the local input buffer.  
    ```c++
    //--- copy data for r2c 3D FFT
    double RealGlobalInput[N1][N2][N3];
    std::vector<double> real_input_buffer;
    fft_mngr.copy_array_into_input_buffer( &( RealGlobalInput[0][0][0] ), real_input_buffer );

    //--- copy data for c2c 3D FFT
    OpenFFT::dcomplex GlobalInput[N1][N2][N3];
    std::vector<OpenFFT::dcomplex> input_buffer;
    fft_mngr.copy_array_into_input_buffer( &( GlobalInput[0][0][0] ), input_buffer );

    //--- copy data for c2c 4D FFT
    OpenFFT::dcomplex GlobalInput[N1][N2][N3][N4];
    std::vector<OpenFFT::dcomplex> input_buffer;
    fft_mngr.copy_array_into_input_buffer( &( GlobalInput[0][0][0][0] ), input_buffer );
    ```
    All 'buffer' input/output accepts both of reference to `std::vector<T>` and pointer to buffer array (`double *` or `OpenFFT::dcomplex *`).  
    Using `std::vector<T>` is recommended because the buffer length is checked for input buffer or resized for output buffer in wrapper API.

  - Step 3: Execute FFT.  
    ```c++
    //--- for r2c 3D FFT
    std::vector<OpenFFT::dcomplex> output_buffer;
    fft_mngr.fft_r2c_forward( real_input_buffer, output_buffer );

    //--- for c2c 3D/4D FFT
    std::vector<OpenFFT::dcomplex> output_buffer;
    fft_mngr.fft_c2c_forward( input_buffer, output_buffer );
    ```

  - Step 4: write back the local output buffer data into the local 3D/4D array.  
    ```c++
    //--- write back for r2c_3D FFT
    OpenFFT::dcomplex LocalOutput[N1][N2][N3r];   // N3r = N3/2+1
    fft_mngr.copy_array_from_output_buffer( &( GlobalOutput[0][0][0] ), output_buffer);

    //--- write back for c2c_3D FFT
    OpenFFT::dcomplex LocalOutput[N1][N2][N3];
    fft_mngr.copy_array_from_output_buffer( &( GlobalOutput[0][0][0] ), output_buffer);

    //--- write back for c2c_4D FFT
    OpenFFT::dcomplex LocalOutput[N1][N2][N3][N4];
    fft_mngr.copy_array_from_output_buffer( &( GlobalOutput[0][0][0][0] ), output_buffer);
    ```

   - Step 4-2: Transpose the output buffer into the input buffer for Inverse FFT (available in c2c_3D or c2c_4D mode only).  
     ```c++
     fft_mngr.transpose_output_to_input( output_buffer, input_buffer );
     ```
     In this function, communicate the data of 'output_buffer' by using MPI_Alltoallv() and reshape received data into the 'input_buffer'.

   - Step 4-3: Execute Inverse FFT (available in c2c_3D or c2c_4D mode only).  
     ```c++
     fft_mngr.fft_c2c_backward( input_buffer, output_buffer );
     ```
     This function is compatible with backward DFT compute of FFTW3.

   - step 5: Finalize.  
     ```c++
     fft_mngr.finalize();
     ```
     The OpenFFT configurations are shared in global manager object (It allows you to make many instance of `OpenFFT::Manager<>`).  
     It is recommended that you call the 'finalize' function at once in finalization part of your program.

## Other API

   - Get OpenFFT information of local process.  
     ```c++
     int                My_Max_NumGrid = fft_mngr.get_max_n_grid();
     int                My_NumGrid_In  = fft_mngr.get_n_grid_in();
     int                My_NumGrid_Out = fft_mngr.get_n_grid_out();
     std::array<int, 8> My_Index_In    = fft_mngr.get_index_in();
     std::array<int, 8> My_Index_Out   = fft_mngr.get_index_out();

     double                elapsed_time = fft_mngr.get_time();
     OpenFFT::FFT_GridType grid_type    = fft_mngr.get_grid_type();
     ```

     The values of [6] and [7] in "My_Index_In" and "My_Index_Out" are significant only at c2c_4D mode.  
     `OpenFFT::FFT_GridType` is enum class type. It is defined as below.
     ```c++
     namespace OpenFFT {
         enum class FFT_GridType {
             none,
             r2c_3D,
             c2c_3D,
             c2c_4D,
         };
     }
     ```

   - Get OpenFFT information of other process.  
     ```c++
     int tgt_proc;  // your target MPI process ID

     int                NumGrid_In  = fft_mngr.get_n_grid_in(  tgt_proc );
     int                NumGrid_Out = fft_mngr.get_n_grid_out( tgt_proc );
     std::array<int, 8> Index_In    = fft_mngr.get_index_in(   tgt_proc );
     std::array<int, 8> Index_Out   = fft_mngr.get_index_out(  tgt_proc );
     ```

   - Gather functions are available to obtain the global 3D/4D array data from local output buffer.  
     ```c++
     //--- gather for r2c 3D FFT and c2c 3D FFT
     OpenFFT::dcomplex GlobalOutput[N1][N2][N3];
     const int tgt_proc = 0; // MPI proc id
     fft_mngr.gather_array( &( GlobalOutput[0][0][0] ), output_buffer, tgt_proc );

     //--- gather for c2c 4D FFT
     OpenFFT::dcomplex GlobalOutput[N1][N2][N3][N4];
     const int tgt_proc = 0; // MPI proc id
     fft_mngr.gather_array( &( GlobalOutput[0][0][0][0] ), output_buffer, tgt_proc );


     //--- allgather for r2c 3D FFT and c2c 3D FFT
     OpenFFT::dcomplex GlobalOutput[N1][N2][N3];
     fft_mngr.allgather_array( &( GlobalOutput[0][0][0] ), output_buffer );

     //--- allgather for c2c 4D FFT
     OpenFFT::dcomplex GlobalOutput[N1][N2][N3][N4];
     fft_mngr.allgather_array( &( GlobalOutput[0][0][0][0] ), output_buffer );



     //--- gather user-defined data type (c2c_3D example)
     struct YourData {
         // define anything your data.
         // (notice: this class will be transported with MPI function.
         //          be careful if this class contains pointer or reference member.)
     };
     std::vector<YoutData> your_output;
     Yourdata YourGlobalOutput[N1][N2][N3];

     fft_mngr.gather_array( &( YourGlobalOutput[0][0][0] ), your_output, tgt_proc );
     ```
     These 'gather' functions accept any type of 3D/4D array and buffer (defined as template class).  
     The gather interface may reduce data transport than using `MPI_Reduce()` or `MPI_Allreduce()` to make global array.

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

     fft_mngr.apply_array_with_output_buffer( &(RealGlobalOutput[0][0][0]), output_buffer, your_func );



     //--- example 2: if you have the local buffer in other process.
     int proc_i;  // MPI process ID which has the part of 'output_buffer_at_proc_i'.
     std::vector<OpenFFT::dcomplex> output_buffer_at_proc_i;

     //------- this API accepts explicit MPI process ID.
     fft_mngr.apply_array_with_output_buffer( &(RealGlobalOutput[0][0][0]), output_buffer_at_proc_i, your_func, proc_i);
     ```
     The implementations of `OpenFFT::Manager<>::apply_array_with_[input/output]_buffer()` are used in other API between global 3D/4D array and local buffer. These 'apply' functions can accept explicit MPI process ID.

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
     const ReduceValue local_sum = fft_mngr.apply_array_with_output_buffer( &(ApplyMatrix[0][0][0]), output_buffer, reduce_value );

     //--- get global sum
     double global_sum;
     MPI_Allreduce( &local_sum.v, &global_sum, 1
                   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     ```
     The apply interface accept any type of 3D/4D array and buffer (both of input are defined as template class respectively).
     ```c++
     //--- perform 'apply' between user-defined class array and buffer.
     struct ArrayClass {
         double v;
         int    m;
     };
     struct BufferClass {
         int j, k;
         OpenFFT::dcomplex c;
     };

     //--- something function between ArrayClass and BufferClass
     struct ApplyFunc {
         void operator () (ArrayClass &arr_v, const BufferClass buf_v){
             arr_v.v =  buf_v.c.r * static_cast<double>(buf_v.j)
                     +  buf_v.c.i * static_cast<double>(buf_v.k);
             arr_v.m = buf_v.j - buf_v.k;
         }
     };
     ApplyFunc apply_func;

     ArrayClass array_3d[N1][N2][N3];
     std::vector<BufferClass> apply_buffer;

     //--- apply your function (select as index pattern)
     fft_mngr.apply_array_with_input_buffer(  &(array_3d[0][0][0]), apply_buffer, apply_func );
     //  or
     fft_mngr.apply_array_with_output_buffer( &(array_3d[0][0][0]), apply_buffer, apply_func );
     ```

   - Index sequence generator for input/output buffer.
     ```c++
     //--- for 3D array
     int tgt_proc = 0;
     std::vector< std::array<int, 3> > index3d_seq;

     fft_mngr.gen_input_index_sequence(index3d_seq);  //  for local proc.
     fft_mngr.gen_output_index_sequence(index3d_seq);

     fft_mngr.gen_input_index_sequence(index3d_seq , tgt_proc);  //  for another proc
     fft_mngr.gen_output_index_sequence(index3d_seq, tgt_proc);

     //--- usage sample for 3D array: make input buffer
     complex_t InputArray3D[N1][N2][N3];
     std::vector<complex_t> input_buffer;
     input_buffer.reserve(My_Max_NumGrid);
     input_buffer.clear();
     for(const auto& g : index3d_seq){
         input_buffer.emplace_back( InputArray3D[ g[0] ][ g[1] ][ g[2] ] );
     }



     //--- for 4D array
     int tgt_proc = 0;
     std::vector< std::array<int, 4> > index4d_seq;

     fft_mngr.gen_input_index_sequence(index4d_seq);  //  for local proc.
     fft_mngr.gen_output_index_sequence(index4d_seq);

     fft_mngr.gen_input_index_sequence(index4d_seq , tgt_proc);  //  for another proc.
     fft_mngr.gen_output_index_sequence(index4d_seq, tgt_proc);

     //--- usage sample for 4D array: copy from output buffer
     complex_t OutputArray4D[N1][N2][N3][N4];
     std::vector<complex_t> output_buffer;
     output_buffer.reserve(My_Max_NumGrid);
     output_buffer.clear();

     fft_mngr.fft_c2c_forward( input_buffer, output_buffer );

     for(size_t ii=0; ii<index4d_seq.size(); ++ii){
         const auto g = index4d_seq[ii];
         OutputArray4D[ g[0] ][ g[1] ][ g[2] ][ g[3] ] = output_buffer[ii];
     }
     ```

     The index sequence generator interface is useful to make the data part in local process.

   - Transpose functions with user-defined data class.
     ```c++
     struct YourData {
         // define anything your data.
         // (notice: this class will be transported with MPI function.
         //          be careful if this class contains pointer or reference member.)
     };

     std::vector<YourData> input_buf, output_buf;

     //--- transpose input_buf -> output buf
     fft_mngr.transpose_input_to_output(input_buf, output_buf);

     //--- transpose output_buf -> input_buf
     fft_mngr.transpose_output_to_input(output_buf, input_buf);
     ```
     The transpose interface accept any type of buffer (defined as template class).

   - Buffer size management interface for gather/transpose API
     ```c++
     struct YourData {
         // define anything your data.
     };

     //
     //  after you called gather or transpose function of OpenFFT::Manager.
     //

     //--- get buffer size
     const size_t sz_buf = fft_mngr.get_buf_size<YourData>();

     //--- resize buffer size
     const size_t sz_buf_new = 4;  // your target size
     fft_mngr.shrink_buf<YourData>(sz_buf_new);
     ```
     The buffer for gather/transpose API will be allocated for each data type.  
     The minimum shrinked buffer size is `sz_min = 4 * sz_buf_new`.

## Note
Makefile configuration example to compile the OpenFFT library.  

```makefile
CC  = mpicc -O3 -fopenmp -I$(YOUR_FFTW_DIR)/include -I./include
LIB = -L$(YOUR_FFTW_DIR)/lib -lfftw3 -lfftw3_omp -lm
FC  = 0
```

The setting example is validated at the environment shown in below.  
  - GCC 6.4
  - OpenMPI 2.1.2
  - FFTW 3.3.7
  - OpenFFT 1.2
