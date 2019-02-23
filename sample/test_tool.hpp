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

    struct Result {
        int n_test    = 0;
        int n_failure = 0;

        void clear(){
            this->n_test    = 0;
            this->n_failure = 0;
        }
        void operator += (const bool result){
            this->n_test    += 1;
            if( !result ){
                this->n_failure += 1;
            }
        }
    };

    template <class T>
    bool check_3d_array(const int n1, const int n2, const int n3,
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
            std::cout << oss.str() << std::flush;
            return false;
        } else {
            oss_green(oss, "        Check done.");
            oss << " All elements are correct\n";
            std::cout << oss.str() << std::flush;
            return true;
        }
    }
    template <class T>
    bool check_buffer(const int  n_grid,
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
            std::cout << oss.str() << std::flush;
            return false;
        } else {
            oss_green(oss, "        Check done.");
            oss << " All elements are correct at proc=" << my_rank << "\n";
            std::cout << oss.str() << std::flush;
            return true;
        }
    }


    template <class T>
    bool check_4d_array(const int n1, const int n2, const int n3, const int n4,
                        const T *array_4d,
                        const T *ref_arr  ){

        int fail_count = 0;
        for(int iw=0; iw<n1; ++iw){
            for(int iz=0; iz<n2; ++iz){
                for(int iy=0; iy<n3; ++iy){
                    for(int ix=0; ix<n4; ++ix){
                        const int pos = iw*(n4*n3*n2)
                                      + iz*(n4*n3)
                                      + iy*(n4)
                                      + ix;

                        const auto elem = array_4d[pos];
                        const auto ref  = ref_arr[pos];
                        if( std::abs(elem.r - ref.r) > 0.001 ||
                            std::abs(elem.i - ref.i) > 0.001   ){
                            print_yellow("ERROR");
                            printf(" array[%d,%d,%d,%d] data=(% 3.3f,% 3.3f), ref=(% 3.3f,% 3.3f)\n",
                                    iw, iz, iy, ix, elem.r, elem.i, ref.r, ref.i);
                            ++fail_count;
                        }
                    }
                }
            }
        }
        if(fail_count == 0){
            print_green("        Check done.");
            printf(     " All elements are correct.\n");
            return true;
        } else {
            print_red(  "        Check failure.");
            printf(     " Some elements are incorrect. failed = %d/%d points.\n",
                   fail_count, n1*n2*n3*n4);
            return false;
        }
    }

    int report(const Result res){
        int myid, numprocs;
        MPI_Comm_rank(MPI_COMM_WORLD,&myid);
        MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

        int final = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
            printf("\n");
            print_green("[test result]\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i_proc=0; i_proc<numprocs; ++i_proc){
            MPI_Barrier(MPI_COMM_WORLD);
            if(myid == i_proc){
                if(res.n_failure <= 0){
                    print_green("  [All tests passed]");
                } else {
                    print_red(  "  [" + std::to_string(res.n_failure)
                                + "/" + std::to_string(res.n_test) + " tests were failed]");
                    final = 1;
                }
                printf(" at proc=%d\n", myid);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        return final;
    }

}
