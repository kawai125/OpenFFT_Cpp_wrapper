/*******************************************************
*    This is the color printer for terminal (using escape secuence).
******************************************************/

#pragma once

#include <string>
#include <iostream>
#include <iomanip>

#include "color_printer.hpp"

#include "openfft.hpp"


namespace TEST {

    constexpr int digit = 3;

    std::ostream& operator << (std::ostream& s, const OpenFFT::dcomplex v){
        s << std::fixed;
        s << "(" << std::setprecision(digit) << v.r << ","
                 << std::setprecision(digit) << v.i << ")";
        return s;
    }

    bool eq_value(const OpenFFT::dcomplex lv, const OpenFFT::dcomplex rv){
        if( std::abs(lv.r - rv.r) > 0.001 ||
            std::abs(lv.i - rv.i) > 0.001   ){
            return false;
        }
        return true;
    }
    bool eq_value(const double lv, const double rv){
        if( std::abs(lv - rv) > 0.001 ){
            return false;
        }
        return true;
    }

    template <class T>
    void check_3d_array(const int n1, const int n2, const int n3,
                        const T *array_3d,
                        const T *ref_arr  ){

        std::ostringstream oss;

        bool check_flag = true;
        for(int iz=0; iz<n1; ++iz){
            for(int iy=0; iy<n2; ++iy){
                for(int ix=0; ix<n3; ++ix){
                    const int pos = iz*(n2*n3) + iy*n3 + ix;

                    const auto elem = array_3d[pos];
                    const auto ref  = ref_arr[pos];
                    if( !eq_value(elem, ref) ){

                        oss_yellow(oss, "ERROR");
                        oss << " array[" << iz << "," << iy << "," << ix << "]:";
                        oss << " data=" << std::setprecision(digit) << elem;
                        oss << " ref="  << std::setprecision(digit) << ref  << "\n";

                        check_flag = false;
                    }
                }
            }
        }
        if( ! check_flag ){
            oss_red(oss,   "        Check failure.");
            oss << " Some elements are incorrect\n";
        } else {
            oss_green(oss, "        Check done.");
            oss << " All elements are correct\n";
        }
        std::cout << oss.str() << std::flush;
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
            if( !eq_value(elem, ref) ){
                oss_yellow(oss, "  ERROR");
                oss << " in buffer[" << ii << "]:"
                    << " data=" << elem
                    << ", ref=" << ref  << "\n";
                check_flag = false;
            }
        }
        if( ! check_flag ){
            oss_red(oss,   "        Check failure.");
            oss << " Some elements are incorrect at proc=" << my_rank << "\n";
        } else {
            oss_green(oss, "        Check done.");
            oss << " All elements are correct at proc=" << my_rank << "\n";
        }
        std::cout << oss.str() << std::flush;
    }

}
