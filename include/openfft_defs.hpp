/**************************************************************************************************/
/**
* @file  openfft_defs.hpp
* @brief C++ wrapper for OpenFFT library.
*
*   OpenFFT library
*   http://www.openmx-square.org/openfft/
*/
/**************************************************************************************************/
#pragma once

#include <string>
#include <iostream>


namespace OpenFFT {

    //--- complex type for OpenFFT
    typedef struct { double r,i; } dcomplex;

    //--- functor sample for buffer manipulator
    struct CopyIntoBuffer{
        template <class Tarr, class Tbuf>
        void operator () (const Tarr &arr_v, Tbuf &buf_v){
            buf_v = arr_v;
        }
    };
    struct CopyFromBuffer{
        template <class Tarr, class Tbuf>
        void operator () (Tarr &arr_v, const Tbuf &buf_v){
            arr_v = buf_v;
        }
    };


    //--- grid type info
    enum class FFT_GridType {
        none,
        r2c_3D,
        c2c_3D,
        c2c_4D,
    };

}


//--- output function as "std::cout << (enum class::value)"
inline std::ostream& operator << (std::ostream& s, const OpenFFT::FFT_GridType e){
    std::string str;
    switch (e) {
        case OpenFFT::FFT_GridType::none:
            str = "none";
        break;

        case OpenFFT::FFT_GridType::r2c_3D:
            str = "r2c_3D";
        break;

        case OpenFFT::FFT_GridType::c2c_3D:
            str = "c2c_3D";
        break;

        case OpenFFT::FFT_GridType::c2c_4D:
            str = "c2c_4D";
        break;

        default:
            throw std::invalid_argument("undefined enum value in FFT_GridType.");
    }
    s << str;
    return s;
}


//--- function interface for C++
extern "C" {
    extern double openfft_init_c2c_3d(int N1, int N2, int N3,
			                          int *My_Max_NumGrid,
			                          int *My_NumGrid_In,  int *My_Index_In,
			                          int *My_NumGrid_Out, int *My_Index_Out,
			                          int offt_measure,
			                          int measure_time, int print_memory);

    extern double openfft_init_r2c_3d(int N1, int N2, int N3,
			                          int *My_Max_NumGrid,
			                          int *My_NumGrid_In,  int *My_Index_In,
			                          int *My_NumGrid_Out, int *My_Index_Out,
			                          int offt_measure,
			                          int measure_time, int print_memory);

    extern double openfft_init_c2c_4d(int N1, int N2, int N3, int N4,
			                          int *My_Max_NumGrid,
			                          int *My_NumGrid_In,  int *My_Index_In,
			                          int *My_NumGrid_Out, int *My_Index_Out,
			                          int offt_measure,
			                          int measure_time, int print_memory);

    extern double openfft_exec_c2c_3d(OpenFFT::dcomplex *Rhor, OpenFFT::dcomplex *Rhok);
    extern double openfft_exec_r2c_3d(         double   *Rhor, OpenFFT::dcomplex *Rhok);
    extern double openfft_exec_c2c_4d(OpenFFT::dcomplex *Rhor, OpenFFT::dcomplex *Rhok);

    extern double openfft_finalize();
    extern void   openfft_dtime(double *time);
}
