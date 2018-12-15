/**************************************************************************************************/
/**
* @file  openfft_global_manager.hpp
* @brief implementation for global manager of OpenFFT library wrapper.
*
*   OpenFFT library
*   http://www.openmx-square.org/openfft/
*/
/**************************************************************************************************/
#pragma once

#include <array>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <exception>

#include "openfft_defs.hpp"
#include "openfft_mpi_tools.hpp"


namespace OpenFFT {

    namespace _impl {

        //--- local functor
        struct GenIndex3D{
            int nz, ny, nx;
            void set_grid(const int nz, const int ny, const int nx){
                this->nz = nz;
                this->ny = ny;
                this->nx = nx;
            }
            int operator () (const int iz, const int iy, const int ix){
                const int pos = iz*(this->nx*this->ny)
                              + iy*(this->nx)
                              + ix;

                #ifndef NDEBUG
                    //--- check without offset
                    if( iz<0 || this->nz<=iz ||
                        iy<0 || this->ny<=iy ||
                        ix<0 || this->nx<=ix   ){
                        std::ostringstream oss;
                        oss << "internal error: invalid index for 3D-array." << "\n"
                            << "   (iz, iy, ix) = ("
                            << iz << ", " << iy << ", " << ix << ")" << "\n"
                            << "    must be in range: (0, 0, 0) to ("
                            << this->nz-1 << ", " << this->ny-1 << ", " << this->nx-1 << ")" << "\n";
                        throw std::out_of_range(oss.str());
                    }
                #endif

                return pos;
            }
        };
        struct GenIndex4D{
            int nw, nz, ny, nx;
            void set_grid(const int nw, const int nz, const int ny, const int nx){
                this->nw = nw;
                this->nz = nz;
                this->ny = ny;
                this->nx = nx;
            }
            int operator () (const int iw, const int iz, const int iy, const int ix){
                const int pos = iw*(this->nx*this->ny*this->nz)
                              + iz*(this->nx*this->ny)
                              + iy*(this->nx)
                              + ix;

                #ifndef NDEBUG
                    //--- check without offset
                    if( iw<0 || this->nw<=iw ||
                        iz<0 || this->nz<=iz ||
                        iy<0 || this->ny<=iy ||
                        ix<0 || this->nx<=ix   ){
                        std::ostringstream oss;
                        oss << "internal error: invalid index for 3D-array." << "\n"
                            << "   (iw, iz, iy, ix) = ("
                            << iw << ", " << iz << ", " << iy << ", " << ix << ")" << "\n"
                            << "    must be in range: (0, 0, 0, 0) to ("
                            << this->nw-1 << ", " << this->nz-1 << ", " << this->ny-1 << ", " << this->nx-1 << ")" << "\n";
                        throw std::out_of_range(oss.str());
                    }
                #endif

                return pos;
            }
        };

        /*
        *  @brief global manager for OpenFFT functions
        */
        template <class Tfloat>
        class GlobalManager{
            using float_t   = Tfloat;
            using complex_t = dcomplex;

        public:
            using float_type   = float_t;
            using complex_type = complex_t;
            using IndexList    = std::array<int, 8>;

        private:
            bool plan_flag               = false;
            bool out_in_convert_flag     = false;
            bool out_in_convert_mpi_flag = false;

            int n_proc  = 0;
            int my_rank = 0;

            FFT_GridType grid_type = FFT_GridType::none;
            int          n_x, n_y, n_z, n_w;

            //--- distributed array range
            int       my_max_n_grid;
            int       my_n_grid_in, my_n_grid_out;
            IndexList my_index_in , my_index_out;

            //--- input & output buffer convert matrix
            std::vector<int>                    n_grid_in_list;
            std::vector<int>                    n_grid_out_list;
            std::vector<IndexList>              index_in_list;
            std::vector<IndexList>              index_out_list;

            struct IndexMark{
                int i_proc, index;
            };
            std::vector<IndexMark>              index_array;
            std::vector<IndexMark>              out_in_convert_matrix;

            std::vector<std::vector<int>>       mpi_send_index;
            std::vector<std::vector<int>>       mpi_recv_index;
            std::vector<std::vector<complex_t>> mpi_send_buf;
            std::vector<std::vector<complex_t>> mpi_recv_buf;

        public:
            GlobalManager() = default;
            ~GlobalManager() noexcept {
                if(this->plan_flag){
                    this->finalize();
                }
            }

            //--- copy is prohibited
            template <class Tf>
            GlobalManager(const GlobalManager<Tf> &rv) = delete;
            template <class Tf>
            GlobalManager& operator = (const GlobalManager<Tf> &rv) = delete;

            //----------------------------------------------------------------------
            //    Initializer
            //----------------------------------------------------------------------
            void init_r2c_3d(const int n_z, const int n_y, const int n_x,
                             const int offt_measure,
                             const int measure_time,
                             const int print_memory ){

                if(this->plan_flag) this->finalize();

                openfft_init_r2c_3d(n_z, n_y, n_x,
                                    &this->my_max_n_grid,
                                    &this->my_n_grid_in , this->my_index_in.data(),
                                    &this->my_n_grid_out, this->my_index_out.data(),
                                    offt_measure, measure_time, print_memory);

                this->grid_type = FFT_GridType::r2c_3D;
                this->n_x = n_x;
                this->n_y = n_y;
                this->n_z = n_z;
                this->n_w = 1;

                this->_set_mpi_proc_info();
                this->_collect_buffer_info();  // enable gather function
                this->plan_flag = true;
            }
            void init_c2c_3d(const int n_z, const int n_y, const int n_x,
                             const int offt_measure,
                             const int measure_time,
                             const int print_memory){

                if(this->plan_flag) this->finalize();

                openfft_init_c2c_3d(n_z, n_y, n_x,
                                    &this->my_max_n_grid,
                                    &this->my_n_grid_in , this->my_index_in.data(),
                                    &this->my_n_grid_out, this->my_index_out.data(),
                                    offt_measure, measure_time, print_memory);

                this->grid_type = FFT_GridType::c2c_3D;
                this->n_x = n_x;
                this->n_y = n_y;
                this->n_z = n_z;
                this->n_w = 1;

                this->_set_mpi_proc_info();
                this->_prepare_3d_out_in_convert(false);  // enable gather and convert function
                this->plan_flag = true;
            }
            void init_c2c_4d(const int n_w, const int n_z, const int n_y, const int n_x,
                             const int offt_measure,
                             const int measure_time,
                             const int print_memory){

                if(this->plan_flag) this->finalize();

                openfft_init_c2c_4d(n_w, n_z, n_y, n_x,
                                    &this->my_max_n_grid,
                                    &this->my_n_grid_in , this->my_index_in.data(),
                                    &this->my_n_grid_out, this->my_index_out.data(),
                                    offt_measure, measure_time, print_memory);

                this->grid_type = FFT_GridType::c2c_4D;
                this->n_x = n_x;
                this->n_y = n_y;
                this->n_z = n_z;
                this->n_w = n_w;

                this->_set_mpi_proc_info();
                this->_collect_buffer_info();  // enable gather function
                this->plan_flag = true;
            }
            void finalize(){
                openfft_finalize();
                this->plan_flag               = false;
                this->out_in_convert_flag     = false;
                this->out_in_convert_mpi_flag = false;
                this->grid_type               = FFT_GridType::none;
            }

            //----------------------------------------------------------------------
            //    infomation getter
            //----------------------------------------------------------------------
            bool is_initialized() const { return this->plan_flag; }

            bool is_same_grid(const int          n_z,
                              const int          n_y,
                              const int          n_x,
                              const FFT_GridType grid_type) const {
                return this->is_same_grid(1, n_z, n_y, n_x, grid_type);
            }
            bool is_same_grid(const int          n_w,
                              const int          n_z,
                              const int          n_y,
                              const int          n_x,
                              const FFT_GridType grid_type) const {
                return (n_w == this->n_w) &&
                       (n_z == this->n_z) &&
                       (n_y == this->n_y) &&
                       (n_x == this->n_x) &&
                       (grid_type == this->grid_type);
            }

            //--- get local proc info
            inline int       get_max_n_grid() const { return this->my_max_n_grid; }
            inline int       get_n_grid_in()  const { return this->my_n_grid_in;  }
            inline int       get_n_grid_out() const { return this->my_n_grid_out; }
            inline IndexList get_index_in()   const { return this->my_index_in;   }
            inline IndexList get_index_out()  const { return this->my_index_out;  }

            //--- get other proc info
            inline int       get_n_grid_in( const int i_proc) const {
                this->_check_rank(i_proc);
                return this->n_grid_in_list[ i_proc];
            }
            inline int       get_n_grid_out(const int i_proc) const {
                this->_check_rank(i_proc);
                return this->n_grid_out_list[i_proc];
            }
            inline IndexList get_index_in(  const int i_proc) const {
                this->_check_rank(i_proc);
                return this->index_in_list[  i_proc];
            }
            inline IndexList get_index_out( const int i_proc) const {
                this->_check_rank(i_proc);
                return this->index_out_list[ i_proc];
            }

            //----------------------------------------------------------------------
            //    FFT & IFFT wrapper
            //----------------------------------------------------------------------
            void fft_r2c_3d_forward(float_t   *input,
                                    complex_t *output){
                this->_check_grid_type(FFT_GridType::r2c_3D);
                openfft_exec_r2c_3d(input, output);
            }

            void fft_c2c_3d_forward(complex_t *input,
                                    complex_t *output){
                this->_check_grid_type(FFT_GridType::c2c_3D);
                openfft_exec_c2c_3d(input, output);
            }
            void fft_c2c_3d_backward(complex_t *input,
                                     complex_t *output){
                 this->_check_grid_type(FFT_GridType::c2c_3D);

                //--- get complex conjugate
                for(int i=0; i<this->my_n_grid_in; ++i){
                    input[i].i = - input[i].i;
                }

                //--- performe forword FFT
                openfft_exec_c2c_3d(input, output);

                //--- get complex conjugate
                //       note: FFT_BACKWARD transforme of FFTW3, it not devide by (NZ * Ny * Zx).
                //             this implementation take compatibility with FFTW3.
                for(int i=0; i<this->my_n_grid_out; ++i){
                    output[i].r =   output[i].r;
                    output[i].i = - output[i].i;
                }
            }
            void fft_c2c_4d_forward(complex_t *input,
                                    complex_t *output){
                this->_check_grid_type(FFT_GridType::c2c_4D);
                openfft_exec_c2c_4d(input, output);
            }
            /*   under developping
            void fft_c2c_4d_backward(complex_t *input,
                                     complex_t *output){
                 this->_check_grid_type(FFT_GridType::c2c_4D);

                //--- get complex conjugate
                for(int i=0; i<this->my_n_grid_in; ++i){
                    input[i].i = - input[i].i;
                }

                //--- performe forword FFT
                openfft_exec_c2c_4d(input, output);

                //--- get complex conjugate and divide by N
                const double N_inv = 1.0/static_cast<double>( this->n_x
                                                             *this->n_y
                                                             *this->n_z
                                                             *this->n_w );
                for(int i=0; i<this->my_n_grid_out; ++i){
                    output[i].r =   output[i].r * N_inv;
                    output[i].i = - output[i].i * N_inv;
                }
            }
            */

            //----------------------------------------------------------------------
            //    input buffer manipulator
            //----------------------------------------------------------------------
            template <class T_3d ,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_3d_array_with_input_buffer(T_3d      *array_3d,
                                                       T_buf     *buffer,
                                                       ApplyFunc  func     ) const {
                return this->_apply_3d_array_with_input_buffer_impl(array_3d, buffer,
                                                                    this->get_n_grid_in(),
                                                                    this->get_index_in(),
                                                                    func );
            }
            template <class T_3d ,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_3d_array_with_input_buffer(      T_3d      *array_3d,
                                                             T_buf     *buffer,
                                                             ApplyFunc  func,
                                                       const int        i_proc   ) const {
                return this->_apply_3d_array_with_input_buffer_impl(array_3d, buffer,
                                                                    this->get_n_grid_in(i_proc),
                                                                    this->get_index_in( i_proc),
                                                                    func );
            }
            template <class T_3d ,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_4d_array_with_input_buffer(T_3d      *array_4d,
                                                       T_buf     *buffer,
                                                       ApplyFunc  func     ) const {
                return this->_apply_4d_array_with_input_buffer_impl(array_4d, buffer,
                                                                    this->get_n_grid_in(),
                                                                    this->get_index_in(),
                                                                    func );
            }
            template <class T_3d ,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_4d_array_with_input_buffer(      T_3d      *array_4d,
                                                             T_buf     *buffer,
                                                             ApplyFunc  func,
                                                       const int        i_proc   ) const {
                return this->_apply_4d_array_with_input_buffer_impl(array_4d, buffer,
                                                                    this->get_n_grid_in(i_proc),
                                                                    this->get_index_in( i_proc),
                                                                    func );
            }

            //----------------------------------------------------------------------
            //    output buffer manipulator
            //----------------------------------------------------------------------
            template <class T_3d ,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_3d_array_with_output_buffer(T_3d      *array_3d,
                                                        T_buf     *buffer,
                                                        ApplyFunc  func     ) const {
                return this->_apply_3d_array_with_output_buffer_impl(array_3d, buffer,
                                                                     this->get_n_grid_out(),
                                                                     this->get_index_out(),
                                                                     func );
            }
            template <class T_3d ,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_3d_array_with_output_buffer(      T_3d      *array_3d,
                                                              T_buf     *buffer,
                                                              ApplyFunc  func,
                                                        const int        i_proc   ) const {
                return this->_apply_3d_array_with_output_buffer_impl(array_3d, buffer,
                                                                     this->get_n_grid_out(i_proc),
                                                                     this->get_index_out( i_proc),
                                                                     func );
            }
            template <class T_3d ,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_4d_array_with_output_buffer(T_3d      *array_4d,
                                                        T_buf     *buffer,
                                                        ApplyFunc  func     ) const {
                return this->_apply_4d_array_with_output_buffer_impl(array_4d, buffer,
                                                                     this->get_n_grid_out(),
                                                                     this->get_index_out(),
                                                                     func );
            }
            template <class T_3d ,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_4d_array_with_output_buffer(      T_3d      *array_4d,
                                                              T_buf     *buffer,
                                                              ApplyFunc  func,
                                                        const int        i_proc   ) const {
                return this->_apply_4d_array_with_output_buffer_impl(array_4d, buffer,
                                                                     this->get_n_grid_out(i_proc),
                                                                     this->get_index_out( i_proc),
                                                                     func );
            }

            //----------------------------------------------------------------------
            //    output_buffer to input_buffer converter
            //----------------------------------------------------------------------
            void convert_output_to_input(      complex_t *input_buf,
                                         const complex_t *output_buf){

                if( this->grid_type != FFT_GridType::c2c_3D ){
                    throw std::logic_error("convert function is implemented for c2c_3D only.");
                }

                if( ! this->out_in_convert_flag ){
                    throw std::logic_error("convert_output_to_input() function is not prepared.");
                }

                if(this->out_in_convert_mpi_flag){
                    //--- convert with MPI Alltoall
                    const int n_proc = this->_get_n_proc();

                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        auto&       send_buf   = this->mpi_send_buf[i_proc];
                        const auto& send_index = this->mpi_send_index[i_proc];

                        send_buf.clear();

                        for(size_t ii=0; ii<send_index.size(); ++ii){
                            const int jj = send_index[ii];
                            send_buf.push_back( output_buf[jj] );
                        }
                    }

                    _mpi::alltoall(this->mpi_send_buf, this->mpi_recv_buf);

                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        const auto& recv_buf   = this->mpi_recv_buf[i_proc];
                        const auto& recv_index = this->mpi_recv_index[i_proc];

                        for(size_t ii=0; ii<recv_index.size(); ++ii){
                            const int jj  = recv_index[ii];
                            input_buf[jj] = recv_buf[ii];
                        }
                    }
                } else {
                    //--- convert in local
                    const int my_rank = this->_get_my_rank();
                    auto& send_buf = this->mpi_send_buf[my_rank];
                    send_buf.clear();

                    const auto& send_index = this->mpi_send_index[my_rank];
                    for(size_t ii=0; ii<send_index.size(); ++ii){
                        const int index = send_index[ii];
                        send_buf.push_back( output_buf[index] );
                    }

                    assert( send_buf.size() == static_cast<size_t>(this->my_n_grid_in) );

                    const auto& recv_index = this->mpi_recv_index[my_rank];

                    assert( send_index.size() == recv_index.size() );

                    for(size_t ii=0; ii<recv_index.size(); ++ii){
                        const int index = recv_index[ii];
                        input_buf[index] = send_buf[ii];
                    }
                }
            }

            //----------------------------------------------------------------------
            //    gather inferface for global 3D-array from output_buffer
            //----------------------------------------------------------------------
            void gather_3d_array(      complex_t *array_3d,
                                 const complex_t *output_buf,
                                 const int        tgt_proc   ){

                //--- make send buffer
                const int n_proc  = this->_get_n_proc();
                const int my_rank = this->_get_my_rank();

                auto& send_buf = this->mpi_send_buf[my_rank];
                send_buf.resize(this->my_n_grid_out);
                for(int ii=0; ii<this->my_n_grid_out; ++ii){
                    send_buf[ii] = output_buf[ii];
                }

                //--- collect output buffers
                _mpi::gather(send_buf, this->mpi_recv_buf, tgt_proc);

                if(my_rank != tgt_proc) return;

                //--- build array_3d
                CopyFromBuffer<complex_t, complex_t> copy_from_buffer;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    const int   n_grid_out = this->get_n_grid_out(i_proc);
                    const auto  index_out  = this->get_index_out( i_proc);
                    const auto& recv_buf   = this->mpi_recv_buf[i_proc];

                    this->_apply_3d_array_with_output_buffer_impl(array_3d,
                                                                  recv_buf.data(),
                                                                  n_grid_out,
                                                                  index_out,
                                                                  copy_from_buffer);
                }
            }
            void allgather_3d_array(      complex_t *array_3d,
                                    const complex_t *output_buf){

                //--- make send buffer
                const int n_proc  = this->_get_n_proc();
                const int my_rank = this->_get_my_rank();

                auto& send_buf = this->mpi_send_buf[my_rank];
                send_buf.resize(this->my_n_grid_out);
                for(int ii=0; ii<this->my_n_grid_out; ++ii){
                    send_buf[ii] = output_buf[ii];
                }

                //--- collect output buffers
                _mpi::allgather(send_buf, this->mpi_recv_buf);

                //--- build array_3d
                CopyFromBuffer<complex_t, complex_t> copy_from_buffer;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    const int   n_grid_out = this->get_n_grid_out(i_proc);
                    const auto  index_out  = this->get_index_out( i_proc);
                    const auto& recv_buf   = this->mpi_recv_buf[i_proc];

                    this->_apply_3d_array_with_output_buffer_impl(array_3d,
                                                                  recv_buf.data(),
                                                                  n_grid_out,
                                                                  index_out,
                                                                  copy_from_buffer);
                }
            }
            void gather_4d_array(      complex_t *array_4d,
                                 const complex_t *output_buf,
                                 const int        tgt_proc   ){

                //--- make send buffer
                const int n_proc  = this->_get_n_proc();
                const int my_rank = this->_get_my_rank();

                auto& send_buf = this->mpi_send_buf[my_rank];
                send_buf.resize(this->my_n_grid_out);
                for(int ii=0; ii<this->my_n_grid_out; ++ii){
                    send_buf[ii] = output_buf[ii];
                }

                //--- collect output buffers
                _mpi::gather(send_buf, this->mpi_recv_buf, tgt_proc);

                if(my_rank != tgt_proc) return;

                //--- build array_4d
                CopyFromBuffer<complex_t, complex_t> copy_from_buffer;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    const int   n_grid_out = this->get_n_grid_out(i_proc);
                    const auto  index_out  = this->get_index_out( i_proc);
                    const auto& recv_buf   = this->mpi_recv_buf[i_proc];

                    this->_apply_4d_array_with_output_buffer_impl(array_4d,
                                                                  recv_buf.data(),
                                                                  n_grid_out,
                                                                  index_out,
                                                                  copy_from_buffer);
                }
            }
            void allgather_4d_array(      complex_t *array_4d,
                                    const complex_t *output_buf){

                //--- make send buffer
                const int n_proc  = this->_get_n_proc();
                const int my_rank = this->_get_my_rank();

                auto& send_buf = this->mpi_send_buf[my_rank];
                send_buf.resize(this->my_n_grid_out);
                for(int ii=0; ii<this->my_n_grid_out; ++ii){
                    send_buf[ii] = output_buf[ii];
                }

                //--- collect output buffers
                _mpi::allgather(send_buf, this->mpi_recv_buf);

                //--- build array_4d
                CopyFromBuffer<complex_t, complex_t> copy_from_buffer;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    const int   n_grid_out = this->get_n_grid_out(i_proc);
                    const auto  index_out  = this->get_index_out( i_proc);
                    const auto& recv_buf   = this->mpi_recv_buf[i_proc];

                    this->_apply_4d_array_with_output_buffer_impl(array_4d,
                                                                  recv_buf.data(),
                                                                  n_grid_out,
                                                                  index_out,
                                                                  copy_from_buffer);
                }
            }

            //--- internal table info (for debug)
            void report_convert_matrix(){
                switch (this->grid_type) {
                    case FFT_GridType::c2c_3D:
                        this->_prepare_3d_out_in_convert(true);
                    break;

                    case FFT_GridType::c2c_4D:
                        _mpi::barrier();
                        if(this->_get_my_rank() == 0){
                            std::cout << "convert function is not implemented for c2c_4D." << std::endl;
                        }
                        _mpi::barrier();
                    break;

                    case FFT_GridType::r2c_3D:
                        _mpi::barrier();
                        if(this->_get_my_rank() == 0){
                            std::cout << "convert function is not implemented for r2c_3D." << std::endl;
                        }
                        _mpi::barrier();
                    break;

                    case FFT_GridType::none:
                        _mpi::barrier();
                        if(this->_get_my_rank() == 0){
                            std::cout << "OpenFFT is not initialized." << std::endl;
                        }
                        _mpi::barrier();
                    break;

                    default:
                        throw std::logic_error("internal error: invalid grid type.");
                }

            }

        private:
            void _set_mpi_proc_info(){
                this->n_proc  = _mpi::get_n_proc();
                this->my_rank = _mpi::get_rank();
            }
            int _get_n_proc()  const { return this->n_proc;  }
            int _get_my_rank() const { return this->my_rank; }
            void _check_rank(const int i) const {
                #ifndef NDEBUG
                    if( i < 0 || this->n_proc <= i){
                        std::ostringstream oss;
                        oss << "invalid process rank: " << i << ", must be in range: [0, " << this->n_proc-1 << "].\n";
                        throw std::invalid_argument(oss.str());
                    }
                #endif
            }

            void _check_grid_type(const FFT_GridType grid_type) const {
                if( this->grid_type != grid_type ){
                    std::ostringstream oss;
                    oss << "OpenFFT: grid type error. not initialized for " << grid_type << " FFT.\n"
                        << "   grid type = " << this->grid_type << "\n";
                    throw std::logic_error(oss.str());
                }
            }

            template <class T_arr,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc _apply_3d_array_with_input_buffer_impl(      T_arr     *array_3d,
                                                                   T_buf     *buffer,
                                                             const int        n_grid_in,
                                                             const IndexList &index_in,
                                                                   ApplyFunc  func      ) const {

                if(n_grid_in <= 0) return func;

                if(this->grid_type != FFT_GridType::r2c_3D &&
                   this->grid_type != FFT_GridType::c2c_3D   ){
                       std::ostringstream oss;
                       oss << "OpenFFT: invalid operation. not initialized for 3D-FFT.\n"
                           << "    grid type = " << this->grid_type << "\n";
                    throw std::logic_error(oss.str());
                }

                GenIndex3D get_3d_index;
                get_3d_index.set_grid(this->n_z, this->n_y, this->n_x);

                int ii = 0;
                if(index_in[0] == index_in[3]){
                    const int iz = index_in[0];
                    for(int iy=index_in[1]; iy<=index_in[4]; ++iy){
                        for(int ix=index_in[2]; ix<=index_in[5]; ++ix){
                            func( array_3d[ get_3d_index(iz, iy, ix) ], buffer[ii] );
                            ++ii;
                        }
                    }
                } else if (index_in[0] < index_in[3]){
                    for(int iz=index_in[0]; iz<=index_in[3]; ++iz){
                        if(iz == index_in[0]){
                            for(int iy=index_in[1]; iy<this->n_y; ++iy){
                                for(int ix=index_in[2]; ix<=index_in[5]; ++ix){
                                    func( array_3d[ get_3d_index(iz, iy, ix) ], buffer[ii] );
                                    ++ii;
                                }
                            }
                        } else if(index_in[0] < iz && iz < index_in[3]){
                            for(int iy=0; iy<this->n_y; ++iy){
                                for(int ix=index_in[2]; ix<=index_in[5]; ++ix){
                                    func( array_3d[ get_3d_index(iz, iy, ix) ], buffer[ii] );
                                    ++ii;
                                }
                            }
                        } else if(iz == index_in[3]){
                            for(int iy=0; iy<=index_in[4]; ++iy){
                                for(int ix=index_in[2]; ix<=index_in[5]; ++ix){
                                    func( array_3d[ get_3d_index(iz, iy, ix) ], buffer[ii] );
                                    ++ii;
                                }
                            }
                        }
                    }
                }

                #ifndef NDEBUG
                    if( ii != n_grid_in ){
                        std::ostringstream oss;
                        oss << "internal array access error." << "\n"
                            << "   buffer size = " << n_grid_in << "\n"
                            << "   n_element   = " << ii << "\n";
                        throw std::logic_error(oss.str());
                    }
                #endif

                return func;
            }
            template <class T_arr,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc _apply_3d_array_with_output_buffer_impl(      T_arr     *array_3d,
                                                                    T_buf     *buffer,
                                                              const int        n_grid_out,
                                                              const IndexList &index_out,
                                                                    ApplyFunc  func       ) const {

                if(n_grid_out <= 0) return func;

                if(this->grid_type != FFT_GridType::r2c_3D &&
                   this->grid_type != FFT_GridType::c2c_3D   ){
                       std::ostringstream oss;
                       oss << "OpenFFT: invalid operation. not initialized for 3D-FFT.\n"
                           << "    grid type = " << this->grid_type << "\n";
                    throw std::logic_error(oss.str());
                }

                GenIndex3D get_3d_index;
                get_3d_index.set_grid(this->n_z, this->n_y, this->n_x);
                if(this->grid_type == FFT_GridType::r2c_3D){
                    get_3d_index.nx = this->n_x/2+1;
                }

                int ii = 0;
                if(index_out[0] == index_out[3]){
                    const int ix = index_out[0];
                    for(int iy=index_out[1]; iy<=index_out[4]; ++iy){
                        for(int iz=index_out[2]; iz<=index_out[5]; ++iz){
                            func( array_3d[ get_3d_index(iz, iy, ix) ], buffer[ii] );
                            ++ii;
                        }
                    }
                } else if(index_out[0] < index_out[3]){
                    for(int ix=index_out[0]; ix<=index_out[3]; ++ix){
                        if(ix == index_out[0]){
                            for(int iy=index_out[1]; iy<this->n_y; ++iy){
                                for(int iz=index_out[2]; iz<=index_out[5]; ++iz){
                                    func( array_3d[ get_3d_index(iz, iy, ix) ], buffer[ii] );
                                    ++ii;
                                }
                            }
                        } else if(index_out[0] < ix && ix < index_out[3]){
                            for(int iy=0; iy<this->n_y; ++iy){
                                for(int iz=index_out[2]; iz<=index_out[5]; ++iz){
                                    func( array_3d[ get_3d_index(iz, iy, ix) ], buffer[ii] );
                                    ++ii;
                                }
                            }
                        } else if (ix == index_out[3]){
                            for(int iy=0; iy<=index_out[4]; ++iy){
                                for(int iz=index_out[2]; iz<=index_out[5]; ++iz){
                                    func( array_3d[ get_3d_index(iz, iy, ix) ], buffer[ii] );
                                    ++ii;
                                }
                            }
                        }
                    }
                }

                #ifndef NDEBUG
                    if( ii != n_grid_out ){
                        std::ostringstream oss;
                        oss << "internal array access error." << "\n"
                            << "   buffer size = " << n_grid_out << "\n"
                            << "   n_element   = " << ii << "\n";
                        throw std::logic_error(oss.str());
                    }
                #endif

                return func;
            }

            template <class T_arr,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc _apply_4d_array_with_input_buffer_impl(      T_arr     *array_4d,
                                                                   T_buf     *buffer,
                                                             const int        n_grid_in,
                                                             const IndexList &index_in,
                                                                   ApplyFunc  func      ) const {

                if(n_grid_in <= 0) return func;

                if(this->grid_type != FFT_GridType::c2c_4D ){
                       std::ostringstream oss;
                       oss << "OpenFFT: invalid operation. not initialized for 4D-FFT.\n"
                           << "    grid type = " << this->grid_type << "\n";
                    throw std::logic_error(oss.str());
                }

                GenIndex4D get_4d_index;
                get_4d_index.set_grid(this->n_w, this->n_z, this->n_y, this->n_x);

                int ii = 0;
                if(index_in[0] == index_in[4]){
                    const int iw = index_in[0];
                    if(index_in[1] == index_in[5]){
                        const int iz = index_in[1];
                        for(int iy=index_in[2]; iy<=index_in[6]; iy++){
                            for(int ix=index_in[3]; ix<=index_in[7]; ix++){
                                func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                ++ii;
                            }
                        }
                    } else if (index_in[1] < index_in[5]){
                        for(int iz=index_in[1]; iz<=index_in[5]; ++iz){
                            if(iz == index_in[1]){
                                for(int iy=index_in[2]; iy<this->n_y; ++iy){
                                    for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                        func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                        ++ii;
                                    }
                                }
                            }
                            else if(index_in[1] < iz && iz < index_in[5]){
                                for(int iy=0; iy<this->n_y; ++iy){
                                    for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                        func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                        ++ii;
                                    }
                                }
                            }
                            else if(iz == index_in[5]){
                                for(int iy=0; iy<=index_in[6]; ++iy){
                                    for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                        func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                        ++ii;
                                    }
                                }
                            }
                        }
                    }
                } else if(index_in[0] < index_in[4]){
                    for(int iw=index_in[0]; iw<=index_in[4]; ++iw){
                        if(iw == index_in[0]){
                            for(int iz=index_in[1]; iz<this->n_z; ++iz){
                                if(iz == index_in[1]){
                                    for(int iy=index_in[2]; iy<this->n_y; ++iy){
                                        for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                            func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                            ++ii;
                                        }
                                    }
                                } else {
                                    for(int iy=0; iy<this->n_y; ++iy){
                                        for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                            func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                            ++ii;
                                        }
                                    }
                                }
                            }
                        } else if(index_in[0] < iw && iw < index_in[4]){
                            for(int iz=0; iz<this->n_z; ++iz){
                                for(int iy=0; iy<this->n_y; ++iy){
                                    for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                        func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                        ++ii;
                                    }
                                }
                            }
                        } else if(iw == index_in[4]){
                            for(int iz=0; iz<=index_in[5]; ++iz){
                                if(iz == index_in[5]){
                                    for(int iy=0; iy<=index_in[6]; ++iy){
                                        for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                            func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                            ++ii;
                                        }
                                    }
                                } else {
                                    for(int iy=0; iy<this->n_y; ++iy){
                                        for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                            func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                            ++ii;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                #ifndef NDEBUG
                    if( ii != n_grid_in ){
                        std::ostringstream oss;
                        oss << "internal array access error." << "\n"
                            << "   buffer size = " << n_grid_in << "\n"
                            << "   n_element   = " << ii << "\n";
                        throw std::logic_error(oss.str());
                    }
                #endif

                return func;
            }
            template <class T_arr,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc _apply_4d_array_with_output_buffer_impl(      T_arr     *array_4d,
                                                                    T_buf     *buffer,
                                                              const int        n_grid_out,
                                                              const IndexList &index_out,
                                                                    ApplyFunc  func       ) const {

                if(n_grid_out <= 0) return func;

                if(this->grid_type != FFT_GridType::c2c_4D ){
                       std::ostringstream oss;
                       oss << "OpenFFT: invalid operation. not initialized for 4D-FFT.\n"
                           << "    grid type = " << this->grid_type << "\n";
                    throw std::logic_error(oss.str());
                }

                GenIndex4D get_4d_index;
                get_4d_index.set_grid(this->n_w, this->n_z, this->n_y, this->n_x);

                int ii = 0;
                if(index_out[0] == index_out[4]){
                    const int ix = index_out[0];
                    if(index_out[1] == index_out[5]){
                        const int iy = index_out[1];
                        for(int iz=index_out[2]; iz<=index_out[6]; iz++){
                            for(int iw=index_out[3]; iw<=index_out[7]; iw++){
                                func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                ++ii;
                            }
                        }
                    } else if (index_out[1] < index_out[5]){
                        for(int iy=index_out[1]; iy<=index_out[5]; ++iy){
                            if(iy == index_out[1]){
                                for(int iz=index_out[2]; iz<this->n_z; ++iz){
                                    for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                        func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                        ++ii;
                                    }
                                }
                            }
                            else if(index_out[1] < iy && iy < index_out[5]){
                                for(int iz=0; iz<this->n_z; ++iz){
                                    for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                        func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                        ++ii;
                                    }
                                }
                            }
                            else if(iy == index_out[5]){
                                for(int iz=0; iz<=index_out[6]; ++iz){
                                    for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                        func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                        ++ii;
                                    }
                                }
                            }
                        }
                    }
                } else if(index_out[0] < index_out[4]){
                    for(int ix=index_out[0]; ix<=index_out[4]; ++ix){
                        if(ix == index_out[0]){
                            for(int iy=index_out[1]; iy<this->n_y; ++iy){
                                if(iy == index_out[1]){
                                    for(int iz=index_out[2]; iz<this->n_z; ++iz){
                                        for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                            func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                            ++ii;
                                        }
                                    }
                                } else {
                                    for(int iz=0; iz<this->n_z; ++iz){
                                        for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                            func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                            ++ii;
                                        }
                                    }
                                }
                            }
                        } else if(index_out[0] < ix && ix < index_out[4]){
                            for(int iy=0; iy<this->n_y; ++iy){
                                for(int iz=0; iz<this->n_z; ++iz){
                                    for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                        func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                        ++ii;
                                    }
                                }
                            }
                        } else if(ix == index_out[4]){
                            for(int iy=0; iy<=index_out[5]; ++iy){
                                if(iy == index_out[5]){
                                    for(int iz=0; iz<=index_out[6]; ++iz){
                                        for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                            func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                            ++ii;
                                        }
                                    }
                                } else {
                                    for(int iz=0; iz<this->n_z; ++iz){
                                        for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                            func( array_4d[ get_4d_index(iw, iz, iy, ix) ], buffer[ii] );
                                            ++ii;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                #ifndef NDEBUG
                    if( ii != n_grid_out ){
                        std::ostringstream oss;
                        oss << "internal array access error." << "\n"
                            << "   buffer size = " << n_grid_out << "\n"
                            << "   n_element   = " << ii << "\n";
                        throw std::logic_error(oss.str());
                    }
                #endif

                return func;
            }

            void _collect_buffer_info(){
                const int n_proc  = this->_get_n_proc();

                //--- collect range info
                _mpi::allgather(this->my_n_grid_in , this->n_grid_in_list );
                _mpi::allgather(this->my_n_grid_out, this->n_grid_out_list);
                _mpi::allgather(this->my_index_in  , this->index_in_list  );
                _mpi::allgather(this->my_index_out , this->index_out_list );

                this->mpi_send_index.resize(n_proc);
                this->mpi_recv_index.resize(n_proc);
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    this->mpi_send_index[i_proc].clear();
                    this->mpi_recv_index[i_proc].clear();
                }

                this->mpi_send_buf.resize(n_proc);
                this->mpi_recv_buf.resize(n_proc);
            }
            void _prepare_3d_out_in_convert(const bool report_matrix = false){
                const int n_proc  = this->_get_n_proc();
                const int my_rank = this->_get_my_rank();

                this->_collect_buffer_info();

                //--- build convert matrix
                this->out_in_convert_matrix.resize( this->n_z * this->n_y * this->n_x );

                //--- trace output -> input
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    const int       n_grid_out = this->n_grid_out_list[i_proc];
                    const IndexList index_out  = this->index_out_list[i_proc];

                    this->index_array.resize(n_grid_out);
                    for(int i=0; i<n_grid_out; ++i){
                        auto& elem = this->index_array[i];
                        elem.i_proc = i_proc;
                        elem.index  = i;
                    }

                    CopyFromBuffer<IndexMark, IndexMark> copy_from_buffer;
                    this->_apply_3d_array_with_output_buffer_impl(this->out_in_convert_matrix.data(),
                                                                  this->index_array.data(),
                                                                  n_grid_out,
                                                                  index_out,
                                                                  copy_from_buffer );
                }

                if(report_matrix){
                    if(my_rank == 0){
                        GenIndex3D get_3d_index;
                        get_3d_index.set_grid(this->n_z, this->n_y, this->n_x);

                        std::ostringstream oss;
                        oss << " == out -> in convert_matrix ==   elem : [source_proc, source_index] " << "\n";
                        for(int iz=0; iz<this->n_z; ++iz){
                            oss << " iz = " << iz << ",\n";
                            for(int iy=0; iy<this->n_y; ++iy){
                                oss << "   (";
                                for(int ix=0; ix<this->n_x; ++ix){
                                    const auto elem = this->out_in_convert_matrix[ get_3d_index(iz, iy, ix) ];
                                    oss << " [" << std::setw(2) << std::right << elem.i_proc << ","
                                                << std::setw(2) << std::right << elem.index  << "]";
                                }
                                oss << " )\n";
                            }
                        }
                        std::cout << oss.str() << std::endl;
                    }
                    _mpi::barrier();
                }

                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    const int       n_grid_in = this->n_grid_in_list[i_proc];
                    const IndexList index_in  = this->index_in_list[i_proc];

                    //--- get source index
                    CopyIntoBuffer<IndexMark, IndexMark> copy_from_buffer;
                    this->index_array.resize(n_grid_in);
                    this->_apply_3d_array_with_input_buffer_impl(this->out_in_convert_matrix.data(),
                                                                 this->index_array.data(),
                                                                 n_grid_in,
                                                                 index_in,
                                                                 copy_from_buffer );

                    if(report_matrix){
                        if(my_rank == 0){
                            std::ostringstream oss;
                            if(i_proc == 0) oss << " == source of input buffer == " << "\n";
                            oss << "  proc = " << i_proc << ", len = " << n_grid_in << "\n";
                            oss << "    (";
                            for(int ii=0; ii<n_grid_in; ++ii){
                                const auto elem = this->index_array[ii];
                                oss << " [" << std::setw(2) << std::right << elem.i_proc << ","
                                            << std::setw(2) << std::right << elem.index  << "]";
                            }
                            oss << " )\n";
                            std::cout << oss.str() << std::flush;
                        }
                        _mpi::barrier();
                    }

                    //--- build send / recv index
                    for(size_t ii=0; ii<this->index_array.size(); ++ii){
                        const int source_proc  = this->index_array[ii].i_proc;
                        const int source_index = this->index_array[ii].index;

                        //--- send index
                        if(source_proc == my_rank){
                            this->mpi_send_index[i_proc].push_back(source_index);
                        }

                        //--- recv index
                        if(i_proc == my_rank){
                            this->mpi_recv_index[source_proc].push_back(ii);
                        }
                    }
                }

                if(report_matrix){
                    _mpi::barrier();
                    if(my_rank == 0) std::cout << "\n";
                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        if(i_proc == my_rank){
                            std::ostringstream oss;
                            if(i_proc == 0) oss << " == mpi_send_index ==    elem : index of output_buffer" << "\n";
                            oss << "   from proc = " << i_proc << ",\n";
                            for(int j_proc=0; j_proc<n_proc; ++j_proc){
                                oss << "      to proc = " << j_proc << ": ";
                                const auto& list = this->mpi_send_index[j_proc];
                                for(const auto& elem : list){
                                    oss << " " << std::setw(2) << std::right << elem;
                                }
                                oss << "\n";
                            }
                            std::cout << oss.str() << std::endl;
                        }
                        _mpi::barrier();
                    }
                    _mpi::barrier();
                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        if(i_proc == my_rank){
                            std::ostringstream oss;
                            if(i_proc == 0) oss << " == mpi_recv_index ==    elem : index of input_buffer" << "\n";
                            oss << "     at proc = " << i_proc << ",\n";
                            for(int j_proc=0; j_proc<n_proc; ++j_proc){
                                oss << "      from proc = " << j_proc << ": ";
                                const auto& list = this->mpi_recv_index[j_proc];
                                for(const auto& elem : list){
                                    oss << " " << std::setw(2) << std::right << elem;
                                }
                                oss << "\n";
                            }
                            std::cout << oss.str() << std::endl;
                        }
                        _mpi::barrier();
                    }
                }

                //--- check send & recv grid size
                int n_grid = 0;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    n_grid += this->mpi_send_index[i_proc].size();
                }
                if( n_grid != this->get_n_grid_out() ){
                    std::ostringstream oss;
                    oss << "internal error: failure to make mpi_send_index\n"
                        << "   total len = " << n_grid
                        << ", must be = " << this->get_n_grid_out() << " (= n_grid_out).\n";
                    throw std::logic_error(oss.str());
                }

                n_grid = 0;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    n_grid += this->mpi_recv_index[i_proc].size();
                }
                if( n_grid != this->get_n_grid_in() ){
                    std::ostringstream oss;
                    oss << "internal error: failure to make mpi_recv_index\n"
                        << "   total len = " << n_grid
                        << ", must be = " << this->get_n_grid_in() << " (= n_grid_in).\n";
                    throw std::logic_error(oss.str());
                }

                //--- check MPI communicate is neccesary or not for convert arrays.
                bool mpi_flag = false;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    if(i_proc == my_rank) continue;
                    if(this->mpi_send_index[i_proc].size() > 0) mpi_flag = true;
                }
                this->out_in_convert_mpi_flag = _mpi::sync_OR(mpi_flag);

                if(report_matrix){
                    _mpi::barrier();
                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        if(i_proc == my_rank){
                            if(this->out_in_convert_mpi_flag){
                                std::cout << " -- convert with MPI Alltoall at proc=" << my_rank << std::endl;
                            } else {
                                std::cout << " -- convert in local at proc=" << my_rank << std::endl;
                            }
                        }
                        _mpi::barrier();
                    }
                }

                //--- preparing for convert was completed
                this->out_in_convert_flag = true;
            }
        };

    }

}
