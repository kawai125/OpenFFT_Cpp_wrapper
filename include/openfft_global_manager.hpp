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
                        oss << "internal error: invalid index for 4D-array." << "\n"
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

        template <class Tarr, class Tbuf, class ApplyFunc>
        struct Apply3DInterface {
            Tarr       *arr_ptr;
            Tbuf       *buf_ptr;
            GenIndex3D  gen_index;
            ApplyFunc   func;

            void operator () (const int iz ,const int iy, const int ix, const int ii){
                this->func( this->arr_ptr[this->gen_index(iz, iy, ix)], this->buf_ptr[ii] );
            }
        };
        template <class Tarr, class Tbuf, class ApplyFunc>
        struct Apply4DInterface {
            Tarr       *arr_ptr;
            Tbuf       *buf_ptr;
            GenIndex4D  gen_index;
            ApplyFunc   func;

            void operator () (const int iw, const int iz ,const int iy, const int ix, const int ii){
                this->func( this->arr_ptr[this->gen_index(iw, iz, iy, ix)], this->buf_ptr[ii] );
            }
        };

        struct DoNothing {
            template <class Tarr, class Tbuf>
            void operator () (const Tarr& v_arr, const Tbuf &v_buf) { return; }
        };

        struct GetIndex3DInterface {
            std::array<int,3>       *arr_ptr;
            const std::array<int,3> *buf_ptr;
            GenIndex3D               gen_index;
            DoNothing                func;

            void operator () (const int iz ,const int iy, const int ix, const int ii){
                this->arr_ptr[ii] = std::array<int,3>{iz, iy, ix};
            }
        };
        struct GetIndex4DInterface {
            std::array<int,4>       *arr_ptr;
            const std::array<int,4> *buf_ptr;
            GenIndex4D               gen_index;
            DoNothing                func;

            void operator () (const int iw, const int iz ,const int iy, const int ix, const int ii){
                this->arr_ptr[ii] = std::array<int,4>{iw, iz, iy, ix};
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
            bool plan_flag                 = false;
            bool transpose_flag            = false;
            bool transpose_out_in_mpi_flag = false;
            bool transpose_in_out_mpi_flag = false;

            FFT_GridType grid_type = FFT_GridType::none;
            int          n_x, n_y, n_z, n_w;

            //--- distributed array range
            int       my_max_n_grid;
            int       my_n_grid_in, my_n_grid_out;
            IndexList my_index_in , my_index_out;

            //--- input & output buffer convert matrix
            std::vector<int>       n_grid_in_list;
            std::vector<int>       n_grid_out_list;
            std::vector<IndexList> index_in_list;
            std::vector<IndexList> index_out_list;

            std::vector<std::vector<int>> transpose_out_in_send_index;
            std::vector<std::vector<int>> transpose_out_in_recv_index;
            std::vector<std::vector<int>> transpose_in_out_send_index;
            std::vector<std::vector<int>> transpose_in_out_recv_index;

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
                this->n_w = 0;

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
                this->n_w = 0;

                this->_collect_buffer_info();   // enable gather function
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

                this->_collect_buffer_info();    // enable gather function
                this->plan_flag = true;
            }
            void finalize(){
                if( this->plan_flag ){
                    openfft_finalize();
                    this->plan_flag                 = false;
                    this->transpose_flag            = false;
                    this->transpose_out_in_mpi_flag = false;
                    this->transpose_in_out_mpi_flag = false;
                    this->grid_type                 = FFT_GridType::none;

                    return;
                }
            }

            //----------------------------------------------------------------------
            //    infomation getter
            //----------------------------------------------------------------------
            bool is_initialized() const { return this->plan_flag; }

            bool is_same_grid(const int          n_z,
                              const int          n_y,
                              const int          n_x,
                              const FFT_GridType grid_type) const {
                return this->is_same_grid(0, n_z, n_y, n_x, grid_type);
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

            double get_time() const {
                double t;
                openfft_dtime(&t);
                return t;
            }

            FFT_GridType get_grid_type() const {
                return this->grid_type;
            }

            void get_grid_size(int &nz, int &ny, int &nx) const {
                this->_check_3d_only();
                nz = this->n_z;
                ny = this->n_y;
                nx = this->n_x;

            }
            void get_grid_size(int &nw, int &nz, int &ny, int &nx) const {
                this->_check_grid_type(FFT_GridType::c2c_4D);
                nw = this->n_w;
                nz = this->n_z;
                ny = this->n_y;
                nx = this->n_x;
            }

            //----------------------------------------------------------------------
            //    FFT & IFFT wrapper
            //----------------------------------------------------------------------
            void fft_r2c_forward(float_t   *input,
                                 complex_t *output){
                this->_check_grid_type(FFT_GridType::r2c_3D);
                openfft_exec_r2c_3d(input, output);
            }

            void fft_c2c_forward(complex_t *input,
                                 complex_t *output){
                switch(this->grid_type){
                    case FFT_GridType::c2c_3D:
                        openfft_exec_c2c_3d(input, output);
                    break;

                    case FFT_GridType::c2c_4D:
                        openfft_exec_c2c_4d(input, output);
                    break;

                    case FFT_GridType::none:
                        throw std::logic_error("OpenFFT: manager is not initialized.");
                    break;

                    default:
                        throw std::logic_error("OpenFFT: invalid grid type.");
                }
            }
            void fft_c2c_backward(complex_t *input,
                                  complex_t *output){

                switch(this->grid_type){
                    case FFT_GridType::c2c_3D:
                    case FFT_GridType::c2c_4D:
                    //--- check passed
                    break;

                    case FFT_GridType::r2c_3D:
                        throw std::logic_error("cannot call fft_c2c_backword() for r2c_3D.");
                    break;

                    case FFT_GridType::none:
                        throw std::logic_error("OpenFFT: manager is not initialized.");
                    break;

                    default:
                        throw std::logic_error("OpenFFT: invalid grid type.");

                }

                //--- get complex conjugate
                for(int i=0; i<this->my_n_grid_in; ++i){
                    input[i].i = - input[i].i;
                }

                //--- performe backword FFT
                switch(this->grid_type){
                    case FFT_GridType::c2c_3D:
                        openfft_exec_c2c_3d(input, output);
                    break;

                    case FFT_GridType::c2c_4D:
                        openfft_exec_c2c_4d(input, output);
                    break;

                    default:
                        throw std::logic_error("OpenFFT: undefined grid type.");
                }

                //--- get complex conjugate
                //       note: FFT_BACKWARD transforme of FFTW3, it not devide by (NZ * Ny * Zx).
                //             this implementation take compatibility with FFTW3.
                for(int i=0; i<this->my_n_grid_out; ++i){
                    output[i].i = - output[i].i;
                }
            }

            //----------------------------------------------------------------------
            //    input buffer manipulator
            //----------------------------------------------------------------------
            template <class T_arr,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_array_with_input_buffer(T_arr     *array,
                                                    T_buf     *buffer,
                                                    ApplyFunc  func   ) const {
                const int my_rank = _mpi::get_rank();
                return this->apply_array_with_input_buffer(array, buffer, func, my_rank);
            }
            template <class T_arr,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_array_with_input_buffer(      T_arr     *array,
                                                          T_buf     *buffer,
                                                          ApplyFunc  func,
                                                    const int        i_proc   ) const {

                switch(this->grid_type){
                    case FFT_GridType::r2c_3D:
                    case FFT_GridType::c2c_3D:
                        return this->_apply_3d_array_with_input_buffer_impl(array, buffer,
                                                                            this->get_n_grid_in(i_proc),
                                                                            this->get_index_in( i_proc),
                                                                            Apply3DInterface<T_arr, T_buf, ApplyFunc>{},
                                                                            func );
                    break;

                    case FFT_GridType::c2c_4D:
                        return this->_apply_4d_array_with_input_buffer_impl(array, buffer,
                                                                            this->get_n_grid_in(i_proc),
                                                                            this->get_index_in( i_proc),
                                                                            Apply4DInterface<T_arr, T_buf, ApplyFunc>{},
                                                                            func );
                    break;

                    case FFT_GridType::none:
                        throw std::logic_error("manager is not initialized.");
                    break;

                    default:
                        throw std::logic_error("invalid grid type.");
                }
            }

            //----------------------------------------------------------------------
            //    output buffer manipulator
            //----------------------------------------------------------------------
            template <class T_arr,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_array_with_output_buffer(      T_arr     *array,
                                                           T_buf     *buffer,
                                                           ApplyFunc  func   ) const {
                const int my_rank = _mpi::get_rank();
                return this->apply_array_with_output_buffer(array, buffer, func, my_rank);
            }
            template <class T_arr,
                      class T_buf,
                      class ApplyFunc >
            ApplyFunc apply_array_with_output_buffer(      T_arr     *array,
                                                           T_buf     *buffer,
                                                           ApplyFunc  func,
                                                     const int        i_proc   ) const {

                switch(this->grid_type){
                    case FFT_GridType::r2c_3D:
                    case FFT_GridType::c2c_3D:
                        return this->_apply_3d_array_with_output_buffer_impl(array, buffer,
                                                                             this->get_n_grid_out(i_proc),
                                                                             this->get_index_out( i_proc),
                                                                             Apply3DInterface<T_arr, T_buf, ApplyFunc>{},
                                                                             func );
                    break;

                    case FFT_GridType::c2c_4D:
                        return this->_apply_4d_array_with_output_buffer_impl(array, buffer,
                                                                             this->get_n_grid_out(i_proc),
                                                                             this->get_index_out( i_proc),
                                                                             Apply4DInterface<T_arr, T_buf, ApplyFunc>{},
                                                                             func );
                    break;

                    case FFT_GridType::none:
                        throw std::logic_error("manager is not initialized.");
                    break;

                    default:
                        throw std::logic_error("invalid grid type.");
                }
            }

            //----------------------------------------------------------------------
            //    index sequence generator
            //----------------------------------------------------------------------
            //--- for 3D array
            void gen_input_index_sequence(std::array<int,3> *index_seq) const {
                const int my_rank = _mpi::get_rank();
                this->gen_input_index_sequence(index_seq, my_rank);
            }
            void gen_input_index_sequence(      std::array<int,3> *index_seq,
                                          const int                i_proc    ) const {
                const auto* dummy_ptr = index_seq;
                this->_apply_3d_array_with_input_buffer_impl(index_seq, dummy_ptr,
                                                             this->get_n_grid_in(i_proc),
                                                             this->get_index_in( i_proc),
                                                             GetIndex3DInterface{},
                                                             DoNothing{} );
            }
            void gen_output_index_sequence(std::array<int,3> *index_seq) const {
                const int my_rank = _mpi::get_rank();
                this->gen_output_index_sequence(index_seq, my_rank);
            }
            void gen_output_index_sequence(      std::array<int,3> *index_seq,
                                           const int                i_proc    ) const {
                const auto* dummy_ptr = index_seq;
                this->_apply_3d_array_with_output_buffer_impl(index_seq, dummy_ptr,
                                                              this->get_n_grid_out(i_proc),
                                                              this->get_index_out( i_proc),
                                                              GetIndex3DInterface{},
                                                              DoNothing{} );
            }

            //--- for 4D array
            void gen_input_index_sequence(std::array<int,4> *index_seq) const {
                const int my_rank = _mpi::get_rank();
                this->gen_input_index_sequence(index_seq, my_rank);
            }
            void gen_input_index_sequence(      std::array<int,4> *index_seq,
                                          const int                i_proc    ) const {
                const auto* dummy_ptr = index_seq;
                this->_apply_4d_array_with_input_buffer_impl(index_seq, dummy_ptr,
                                                             this->get_n_grid_in(i_proc),
                                                             this->get_index_in( i_proc),
                                                             GetIndex4DInterface{},
                                                             DoNothing{} );
            }
            void gen_output_index_sequence(std::array<int,4> *index_seq) const {
                const int my_rank = _mpi::get_rank();
                this->gen_output_index_sequence(index_seq, my_rank);
            }
            void gen_output_index_sequence(      std::array<int,4> *index_seq,
                                           const int                i_proc    ) const {
                const auto* dummy_ptr = index_seq;
                this->_apply_4d_array_with_output_buffer_impl(index_seq, dummy_ptr,
                                                              this->get_n_grid_out(i_proc),
                                                              this->get_index_out( i_proc),
                                                              GetIndex4DInterface{},
                                                              DoNothing{} );
            }

            //----------------------------------------------------------------------
            //    transposer between output_buffer and input_buffer
            //----------------------------------------------------------------------
            template <class Tdata>
            void transpose_input_to_output(const Tdata *input_buf,
                                                 Tdata *output_buf){

                this->_setup_transpose();

                if(this->transpose_in_out_mpi_flag){
                    //--- transpose with MPI Alltoall
                    const int n_proc = _mpi::get_n_proc();
                    _mpi::CommImpl<Tdata> comm_impl;
                    comm_impl.init(n_proc);

                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        auto&       send_buf   = comm_impl.get_send_buf(i_proc);
                        const auto& send_index = this->transpose_in_out_send_index[i_proc];

                        send_buf.clear();

                        for(size_t ii=0; ii<send_index.size(); ++ii){
                            const int jj = send_index[ii];
                            send_buf.emplace_back( input_buf[jj] );
                        }
                    }

                    comm_impl.alltoall();

                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        const auto& recv_buf   = comm_impl.get_recv_buf(i_proc);
                        const auto& recv_index = this->transpose_in_out_recv_index[i_proc];

                        for(size_t ii=0; ii<recv_index.size(); ++ii){
                            const int jj   = recv_index[ii];
                            output_buf[jj] = recv_buf[ii];
                        }
                    }
                } else {
                    //--- transpose in local
                    const int n_proc  = _mpi::get_n_proc();
                    const int my_rank = _mpi::get_rank();
                    _mpi::CommImpl<Tdata> comm_impl;
                    comm_impl.init(n_proc);

                    auto& send_buf = comm_impl.get_send_buf(my_rank);
                    send_buf.clear();

                    const auto& send_index = this->transpose_in_out_send_index[my_rank];

                    for(size_t ii=0; ii<send_index.size(); ++ii){
                        const int index = send_index[ii];
                        send_buf.emplace_back( input_buf[index] );
                    }

                    if( send_buf.size() != static_cast<size_t>(this->my_n_grid_out) ){
                        std::ostringstream oss;
                        oss << "internal error: length of 'send_buf' and 'my_n_grid_out' is not match.\n"
                            << "   send_buf.size() = " << send_buf.size()
                            << ", must be = " << this->get_n_grid_out() << " (= n_grid_out).\n";
                        throw std::logic_error(oss.str());
                    }

                    const auto& recv_index = this->transpose_in_out_recv_index[my_rank];

                    if( send_index.size() != recv_index.size() ){
                        std::ostringstream oss;
                        oss << "internal error: length of 'send_index' and 'recv_index' is not match.\n"
                            << "   send_index.size() = " << send_index.size() << "\n"
                            << "   recv_index.size() = " << recv_index.size() << "\n";
                        throw std::logic_error(oss.str());
                    }

                    for(size_t ii=0; ii<recv_index.size(); ++ii){
                        const int index   = recv_index[ii];
                        output_buf[index] = send_buf[ii];
                    }
                }
            }

            template <class Tdata>
            void transpose_output_to_input(const Tdata *output_buf,
                                                 Tdata *input_buf ){

                this->_setup_transpose();

                if(this->transpose_out_in_mpi_flag){
                    //--- transpose with MPI Alltoall
                    const int n_proc = _mpi::get_n_proc();
                    _mpi::CommImpl<Tdata> comm_impl;
                    comm_impl.init(n_proc);

                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        auto&       send_buf   = comm_impl.get_send_buf(i_proc);
                        const auto& send_index = this->transpose_out_in_send_index[i_proc];

                        send_buf.clear();

                        for(size_t ii=0; ii<send_index.size(); ++ii){
                            const int jj = send_index[ii];
                            send_buf.emplace_back( output_buf[jj] );
                        }
                    }

                    comm_impl.alltoall();

                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        const auto& recv_buf   = comm_impl.get_recv_buf(i_proc);
                        const auto& recv_index = this->transpose_out_in_recv_index[i_proc];

                        for(size_t ii=0; ii<recv_index.size(); ++ii){
                            const int jj  = recv_index[ii];
                            input_buf[jj] = recv_buf[ii];
                        }
                    }
                } else {
                    //--- transpose in local
                    const int n_proc  = _mpi::get_n_proc();
                    const int my_rank = _mpi::get_rank();
                    _mpi::CommImpl<Tdata> comm_impl;
                    comm_impl.init(n_proc);

                    auto& send_buf = comm_impl.get_send_buf(my_rank);
                    send_buf.clear();

                    const auto& send_index = this->transpose_out_in_send_index[my_rank];

                    for(size_t ii=0; ii<send_index.size(); ++ii){
                        const int index = send_index[ii];
                        send_buf.emplace_back( output_buf[index] );
                    }

                    if( send_buf.size() != static_cast<size_t>(this->my_n_grid_in) ){
                        std::ostringstream oss;
                        oss << "internal error: length of 'send_buf' and 'my_n_grid_in' is not match.\n"
                            << "   send_buf.size() = " << send_buf.size()
                            << ", must be = " << this->get_n_grid_in() << " (= n_grid_in).\n";
                        throw std::logic_error(oss.str());
                    }

                    const auto& recv_index = this->transpose_out_in_recv_index[my_rank];

                    if( send_index.size() != recv_index.size() ){
                        std::ostringstream oss;
                        oss << "internal error: length of 'send_index' and 'recv_index' is not match.\n"
                            << "   send_index.size() = " << send_index.size() << "\n"
                            << "   recv_index.size() = " << recv_index.size() << "\n";
                        throw std::logic_error(oss.str());
                    }

                    for(size_t ii=0; ii<recv_index.size(); ++ii){
                        const int index = recv_index[ii];
                        input_buf[index] = send_buf[ii];
                    }
                }
            }

            //----------------------------------------------------------------------
            //    gather inferface for global 3D/4D-array from output_buffer
            //----------------------------------------------------------------------
            template <class Tdata>
            void gather_array(      Tdata *array,
                              const Tdata *output_buf,
                              const int    tgt_proc   ){

                //--- make send buffer
                const int n_proc  = _mpi::get_n_proc();
                const int my_rank = _mpi::get_rank();

                _mpi::CommImpl<Tdata> comm_impl;
                comm_impl.gather(output_buf, this->my_n_grid_out, tgt_proc);

                if(my_rank != tgt_proc) return;

                //--- build array
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    const auto& recv_buf = comm_impl.get_recv_buf(i_proc);
                    this->apply_array_with_output_buffer(array, recv_buf.data(), CopyFromBuffer{}, i_proc);
                }
            }
            template <class Tdata>
            void allgather_array(      Tdata *array,
                                 const Tdata *output_buf){

                //--- make send buffer
                const int n_proc  = _mpi::get_n_proc();

                _mpi::CommImpl<Tdata> comm_impl;
                comm_impl.allgather(output_buf, this->my_n_grid_out);

                //--- build array
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    const auto& recv_buf = comm_impl.get_recv_buf(i_proc);
                    this->apply_array_with_output_buffer(array, recv_buf.data(), CopyFromBuffer{}, i_proc);
                }
            }

        private:
            void _check_rank(const int i) const {
                #ifndef NDEBUG
                    if( i < 0 || _mpi::get_n_proc() <= i){
                        std::ostringstream oss;
                        oss << "invalid process rank: " << i << ", must be in range: [0, " << _mpi::get_n_proc()-1 << "].\n";
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
            void _check_c2c_only() const {
                if( this->grid_type != FFT_GridType::c2c_3D &&
                    this->grid_type != FFT_GridType::c2c_4D   ){
                    std::ostringstream oss;
                    oss << "OpenFFT: grid type error. not initialized for c2c type FFT.\n"
                        << "   grid type = " << this->grid_type << "\n";
                    throw std::logic_error(oss.str());
                }
            }
            void _check_3d_only() const {
                if(this->grid_type != FFT_GridType::r2c_3D &&
                   this->grid_type != FFT_GridType::c2c_3D   ){
                       std::ostringstream oss;
                       oss << "OpenFFT: grid type error. not initialized for 3D-FFT.\n"
                           << "    grid type = " << this->grid_type << "\n";
                    throw std::logic_error(oss.str());
                }
            }
            void _setup_transpose(){
                this->_check_c2c_only();

                if( ! this->transpose_flag ){
                    switch (this->grid_type){
                        case FFT_GridType::c2c_3D:
                            this->_prepare_transpose<3>();
                        break;

                        case FFT_GridType::c2c_4D:
                            this->_prepare_transpose<4>();
                        break;

                        default:
                            throw std::logic_error("invalid grid type for transpose function.");
                    }
                }
            }

            template <class Tptr>
            void _check_nullptr(const Tptr *ptr) const {
                if(ptr == nullptr){
                    throw std::invalid_argument("the nullptr was passed.");
                }
            }

            template < class T_arr,
                       class T_buf,
                       class ApplyInterface,
                       class ApplyFunc      >
            ApplyFunc _apply_3d_array_with_input_buffer_impl(      T_arr          *array_3d,
                                                                   T_buf          *buffer,
                                                             const int             n_grid_in,
                                                             const IndexList      &index_in,
                                                                   ApplyInterface  apply_interface,
                                                                   ApplyFunc       apply_func      ) const {

                if(n_grid_in <= 0) return apply_func;

                this->_check_3d_only();

                this->_check_nullptr(array_3d);
                this->_check_nullptr(buffer);

                apply_interface.arr_ptr = array_3d;
                apply_interface.buf_ptr = buffer;
                apply_interface.gen_index.set_grid(this->n_z, this->n_y, this->n_x);
                apply_interface.func    = apply_func;

                int ii = 0;
                if(index_in[0] == index_in[3]){
                    const int iz = index_in[0];
                    for(int iy=index_in[1]; iy<=index_in[4]; ++iy){
                        for(int ix=index_in[2]; ix<=index_in[5]; ++ix){
                            apply_interface(iz, iy, ix, ii);
                            ++ii;
                        }
                    }
                } else if (index_in[0] < index_in[3]){
                    for(int iz=index_in[0]; iz<=index_in[3]; ++iz){
                        if(iz == index_in[0]){
                            for(int iy=index_in[1]; iy<this->n_y; ++iy){
                                for(int ix=index_in[2]; ix<=index_in[5]; ++ix){
                                    apply_interface(iz, iy, ix, ii);
                                    ++ii;
                                }
                            }
                        } else if(index_in[0] < iz && iz < index_in[3]){
                            for(int iy=0; iy<this->n_y; ++iy){
                                for(int ix=index_in[2]; ix<=index_in[5]; ++ix){
                                    apply_interface(iz, iy, ix, ii);
                                    ++ii;
                                }
                            }
                        } else if(iz == index_in[3]){
                            for(int iy=0; iy<=index_in[4]; ++iy){
                                for(int ix=index_in[2]; ix<=index_in[5]; ++ix){
                                    apply_interface(iz, iy, ix, ii);
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

                return apply_interface.func;
            }
            template < class T_arr,
                       class T_buf,
                       class ApplyInterface,
                       class ApplyFunc      >
            ApplyFunc _apply_3d_array_with_output_buffer_impl(      T_arr          *array_3d,
                                                                    T_buf          *buffer,
                                                              const int             n_grid_out,
                                                              const IndexList      &index_out,
                                                                    ApplyInterface  apply_interface,
                                                                    ApplyFunc       apply_func      ) const {

                if(n_grid_out <= 0) return apply_func;

                this->_check_3d_only();

                this->_check_nullptr(array_3d);
                this->_check_nullptr(buffer);

                apply_interface.arr_ptr = array_3d;
                apply_interface.buf_ptr = buffer;
                apply_interface.func    = apply_func;
                if(this->grid_type == FFT_GridType::r2c_3D){
                    apply_interface.gen_index.set_grid(this->n_z, this->n_y, this->n_x/2+1);
                } else {
                    apply_interface.gen_index.set_grid(this->n_z, this->n_y, this->n_x);
                }

                int ii = 0;
                if(index_out[0] == index_out[3]){
                    const int ix = index_out[0];
                    for(int iy=index_out[1]; iy<=index_out[4]; ++iy){
                        for(int iz=index_out[2]; iz<=index_out[5]; ++iz){
                            apply_interface(iz, iy, ix, ii);
                            ++ii;
                        }
                    }
                } else if(index_out[0] < index_out[3]){
                    for(int ix=index_out[0]; ix<=index_out[3]; ++ix){
                        if(ix == index_out[0]){
                            for(int iy=index_out[1]; iy<this->n_y; ++iy){
                                for(int iz=index_out[2]; iz<=index_out[5]; ++iz){
                                    apply_interface(iz, iy, ix, ii);
                                    ++ii;
                                }
                            }
                        } else if(index_out[0] < ix && ix < index_out[3]){
                            for(int iy=0; iy<this->n_y; ++iy){
                                for(int iz=index_out[2]; iz<=index_out[5]; ++iz){
                                    apply_interface(iz, iy, ix, ii);
                                    ++ii;
                                }
                            }
                        } else if (ix == index_out[3]){
                            for(int iy=0; iy<=index_out[4]; ++iy){
                                for(int iz=index_out[2]; iz<=index_out[5]; ++iz){
                                    apply_interface(iz, iy, ix, ii);
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

                return apply_interface.func;
            }

            template < class T_arr,
                       class T_buf,
                       class ApplyInterface,
                       class ApplyFunc      >
            ApplyFunc _apply_4d_array_with_input_buffer_impl(      T_arr          *array_4d,
                                                                   T_buf          *buffer,
                                                             const int             n_grid_in,
                                                             const IndexList      &index_in,
                                                                   ApplyInterface  apply_interface,
                                                                   ApplyFunc       apply_func      ) const {

                if(n_grid_in <= 0) return apply_func;

                this->_check_grid_type(FFT_GridType::c2c_4D);

                this->_check_nullptr(array_4d);
                this->_check_nullptr(buffer);

                apply_interface.arr_ptr = array_4d;
                apply_interface.buf_ptr = buffer;
                apply_interface.gen_index.set_grid(this->n_w, this->n_z, this->n_y, this->n_x);
                apply_interface.func    = apply_func;

                int ii = 0;
                if(index_in[0] == index_in[4]){
                    const int iw = index_in[0];
                    if(index_in[1] == index_in[5]){
                        const int iz = index_in[1];
                        for(int iy=index_in[2]; iy<=index_in[6]; iy++){
                            for(int ix=index_in[3]; ix<=index_in[7]; ix++){
                                apply_interface(iw, iz, iy, ix, ii);
                                ++ii;
                            }
                        }
                    } else if (index_in[1] < index_in[5]){
                        for(int iz=index_in[1]; iz<=index_in[5]; ++iz){
                            if(iz == index_in[1]){
                                for(int iy=index_in[2]; iy<this->n_y; ++iy){
                                    for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                        apply_interface(iw, iz, iy, ix, ii);
                                        ++ii;
                                    }
                                }
                            }
                            else if(index_in[1] < iz && iz < index_in[5]){
                                for(int iy=0; iy<this->n_y; ++iy){
                                    for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                        apply_interface(iw, iz, iy, ix, ii);
                                        ++ii;
                                    }
                                }
                            }
                            else if(iz == index_in[5]){
                                for(int iy=0; iy<=index_in[6]; ++iy){
                                    for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                        apply_interface(iw, iz, iy, ix, ii);
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
                                            apply_interface(iw, iz, iy, ix, ii);
                                            ++ii;
                                        }
                                    }
                                } else {
                                    for(int iy=0; iy<this->n_y; ++iy){
                                        for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                            apply_interface(iw, iz, iy, ix, ii);
                                            ++ii;
                                        }
                                    }
                                }
                            }
                        } else if(index_in[0] < iw && iw < index_in[4]){
                            for(int iz=0; iz<this->n_z; ++iz){
                                for(int iy=0; iy<this->n_y; ++iy){
                                    for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                        apply_interface(iw, iz, iy, ix, ii);
                                        ++ii;
                                    }
                                }
                            }
                        } else if(iw == index_in[4]){
                            for(int iz=0; iz<=index_in[5]; ++iz){
                                if(iz == index_in[5]){
                                    for(int iy=0; iy<=index_in[6]; ++iy){
                                        for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                            apply_interface(iw, iz, iy, ix, ii);
                                            ++ii;
                                        }
                                    }
                                } else {
                                    for(int iy=0; iy<this->n_y; ++iy){
                                        for(int ix=index_in[3]; ix<=index_in[7]; ++ix){
                                            apply_interface(iw, iz, iy, ix, ii);
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

                return apply_interface.func;
            }
            template < class T_arr,
                       class T_buf,
                       class ApplyInterface,
                       class ApplyFunc      >
            ApplyFunc _apply_4d_array_with_output_buffer_impl(      T_arr          *array_4d,
                                                                    T_buf          *buffer,
                                                              const int             n_grid_out,
                                                              const IndexList      &index_out,
                                                                    ApplyInterface  apply_interface,
                                                                    ApplyFunc       apply_func      ) const {

                if(n_grid_out <= 0) return apply_func;

                this->_check_grid_type(FFT_GridType::c2c_4D);

                this->_check_nullptr(array_4d);
                this->_check_nullptr(buffer);

                apply_interface.arr_ptr = array_4d;
                apply_interface.buf_ptr = buffer;
                apply_interface.gen_index.set_grid(this->n_w, this->n_z, this->n_y, this->n_x);
                apply_interface.func    = apply_func;

                int ii = 0;
                if(index_out[0] == index_out[4]){
                    const int ix = index_out[0];
                    if(index_out[1] == index_out[5]){
                        const int iy = index_out[1];
                        for(int iz=index_out[2]; iz<=index_out[6]; iz++){
                            for(int iw=index_out[3]; iw<=index_out[7]; iw++){
                                apply_interface(iw, iz, iy, ix, ii);
                                ++ii;
                            }
                        }
                    } else if (index_out[1] < index_out[5]){
                        for(int iy=index_out[1]; iy<=index_out[5]; ++iy){
                            if(iy == index_out[1]){
                                for(int iz=index_out[2]; iz<this->n_z; ++iz){
                                    for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                        apply_interface(iw, iz, iy, ix, ii);
                                        ++ii;
                                    }
                                }
                            }
                            else if(index_out[1] < iy && iy < index_out[5]){
                                for(int iz=0; iz<this->n_z; ++iz){
                                    for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                        apply_interface(iw, iz, iy, ix, ii);
                                        ++ii;
                                    }
                                }
                            }
                            else if(iy == index_out[5]){
                                for(int iz=0; iz<=index_out[6]; ++iz){
                                    for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                        apply_interface(iw, iz, iy, ix, ii);
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
                                            apply_interface(iw, iz, iy, ix, ii);
                                            ++ii;
                                        }
                                    }
                                } else {
                                    for(int iz=0; iz<this->n_z; ++iz){
                                        for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                            apply_interface(iw, iz, iy, ix, ii);
                                            ++ii;
                                        }
                                    }
                                }
                            }
                        } else if(index_out[0] < ix && ix < index_out[4]){
                            for(int iy=0; iy<this->n_y; ++iy){
                                for(int iz=0; iz<this->n_z; ++iz){
                                    for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                        apply_interface(iw, iz, iy, ix, ii);
                                        ++ii;
                                    }
                                }
                            }
                        } else if(ix == index_out[4]){
                            for(int iy=0; iy<=index_out[5]; ++iy){
                                if(iy == index_out[5]){
                                    for(int iz=0; iz<=index_out[6]; ++iz){
                                        for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                            apply_interface(iw, iz, iy, ix, ii);
                                            ++ii;
                                        }
                                    }
                                } else {
                                    for(int iz=0; iz<this->n_z; ++iz){
                                        for(int iw=index_out[3]; iw<=index_out[7]; ++iw){
                                            apply_interface(iw, iz, iy, ix, ii);
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

                return apply_interface.func;
            }

            bool _is_input_range(const std::array<int, 3> tgt, const IndexList index_in) const {
                if(index_in[0] == index_in[3]){
                    if( tgt[0] == index_in[0] ){
                        if( index_in[1]<=tgt[1] && tgt[1]<=index_in[4] ){
                            if( index_in[2]<=tgt[2] && tgt[2]<=index_in[5] ) return true;
                        }
                    }
                } else if (index_in[0] < index_in[3]){
                    if( tgt[0] == index_in[0] ){
                        if( index_in[1]<=tgt[1] && tgt[1]<this->n_y ){
                            if( index_in[2]<=tgt[2] && tgt[2]<=index_in[5] ) return true;
                        }
                    } else if( index_in[0]<=tgt[0] && tgt[0]<=index_in[3] ){
                        if( 0<=tgt[1] && tgt[1]<this->n_y ){
                            if( index_in[2]<=tgt[2] && tgt[2]<=index_in[5] ) return true;
                        }
                    } else if( tgt[0] == index_in[3] ){
                        if( 0<=tgt[1] && tgt[1]<=index_in[4] ){
                            if( index_in[2]<=tgt[2] && tgt[2]<=index_in[5] ) return true;
                        }
                    }
                }
                return false;
            }
            bool _is_output_range(const std::array<int, 3> tgt, const IndexList index_out) const {
                if(index_out[0] == index_out[3]){
                    if( tgt[2] == index_out[0] ){
                        if( index_out[1]<=tgt[1] && tgt[1]<=index_out[4] ){
                            if( index_out[2]<=tgt[0] && tgt[0]<=index_out[5] ) return true;
                        }
                    }
                } else if(index_out[0] < index_out[3]){
                    if( tgt[2] == index_out[0] ){
                        if( index_out[1]<=tgt[1] && tgt[1]<this->n_y ){
                            if( index_out[2]<=tgt[0] && tgt[0]<=index_out[5] ) return true;
                        }
                    } else if( index_out[0]<tgt[2] && tgt[2]<index_out[3] ){
                        if( 0<=tgt[1] && tgt[1]<this->n_y ){
                            if( index_out[2]<=tgt[0] && tgt[0]<=index_out[5] ) return true;
                        }
                    } else if( tgt[2] == index_out[3] ){
                        if( 0<=tgt[1] && tgt[1]<=index_out[4] ){
                            if( index_out[2]<=tgt[0] && tgt[0]<=index_out[5] ) return true;
                        }
                    }
                }
                return false;
            }

            bool _is_input_range(const std::array<int, 4> tgt, const IndexList index_in) const {
                if(index_in[0] == index_in[4]){
                    if( tgt[0] == index_in[0] ){
                        if( index_in[1] == index_in[5] ){
                            if( tgt[1] == index_in[1] ){
                                if( index_in[2]<=tgt[2] && tgt[2]<=index_in[6] ){
                                    if( index_in[3]<=tgt[3] && tgt[3]<=index_in[7] ) return true;
                                }
                            }
                        } else if(index_in[1] < index_in[5]){
                            if( tgt[1] == index_in[1] ){
                                if( index_in[2]<=tgt[2] && tgt[2]<this->n_y ){
                                    if( index_in[3]<=tgt[3] && tgt[3]<=index_in[7] ) return true;
                                }
                            } else if( index_in[1]<tgt[1] && tgt[1]<index_in[5] ){
                                if( 0<=tgt[2] && tgt[2]<this->n_y ){
                                    if( index_in[3]<=tgt[3] && tgt[3]<=index_in[7] ) return true;
                                }
                            } else if( tgt[1] == index_in[5] ){
                                if( 0<=tgt[2] && tgt[2]<=index_in[6] ){
                                    if( index_in[3]<=tgt[3] && tgt[3]<=index_in[7] ) return true;
                                }
                            }
                        }
                    }
                } else if(index_in[0] < index_in[4]){
                    if( tgt[0] == index_in[0] ){
                        if( tgt[1] == index_in[1] ){
                            if( index_in[2]<=tgt[2] && tgt[2]<this->n_y ){
                                if( index_in[3]<=tgt[3] && tgt[3]<=index_in[7] ) return true;
                            }
                        } else if( index_in[1]<tgt[1] && tgt[1]<this->n_z ){
                            if( 0<=tgt[2] && tgt[2]<this->n_y ){
                                if( index_in[3]<=tgt[3] && tgt[3]<=index_in[7] ) return true;
                            }
                        }
                    } else if( index_in[0]<tgt[0] && tgt[0]<index_in[4] ){
                        if( 0<=tgt[1] && tgt[1]<this->n_z ){
                            if( 0<=tgt[2] && tgt[2]<this->n_y ){
                                if( index_in[3]<=tgt[3] && tgt[3]<=index_in[7] ) return true;
                            }
                        }
                    } else if( tgt[0] == index_in[4] ){
                        if( tgt[1] == index_in[5] ){
                            if( 0<=tgt[2] && tgt[2]<=index_in[6] ){
                                if( index_in[3]<=tgt[3] && tgt[3]<=index_in[7] ) return true;
                            }
                        } else if( 0<=tgt[1] && tgt[1]<index_in[5] ){
                            if( 0<=tgt[2] && tgt[2]<this->n_y ){
                                if( index_in[3]<=tgt[3] && tgt[3]<=index_in[7] ) return true;
                            }
                        }
                    }
                }
                return false;
            }
            bool _is_output_range(const std::array<int, 4> tgt, const IndexList index_out) const {
                if(index_out[0] == index_out[4]){
                    if( tgt[3] == index_out[0] ){
                        if( index_out[1] == index_out[5] ){
                            if( tgt[2] == index_out[1] ){
                                if( index_out[2]<=tgt[1] && tgt[1]<=index_out[6] ){
                                    if( index_out[3]<=tgt[0] && tgt[0]<=index_out[7] ) return true;
                                }
                            }
                        } else if( index_out[1] < index_out[5] ){
                            if( tgt[2] == index_out[1] ){
                                if( index_out[2]<=tgt[1] && tgt[1]<this->n_z ){
                                    if( index_out[3]<=tgt[0] && tgt[0]<=index_out[7] ) return true;
                                }
                            } else if( index_out[1]<tgt[2] && tgt[2]<index_out[5] ){
                                if( 0<=tgt[1] && tgt[1]<this->n_z ){
                                    if( index_out[3]<=tgt[0] && tgt[0]<=index_out[7] ) return true;
                                }
                            } else if( tgt[2] == index_out[5] ){
                                if( 0<=tgt[1] && tgt[1]<=index_out[6] ){
                                    if( index_out[3]<=tgt[0] && tgt[0]<=index_out[7] ) return true;
                                }
                            }
                        }
                    }
                } else if(index_out[0] < index_out[4]){
                    if( tgt[3] == index_out[0] ){
                        if( tgt[2] == index_out[1] ){
                            if( index_out[2]<=tgt[1] && tgt[1]<this->n_z ){
                                if( index_out[3]<=tgt[0] && tgt[0]<=index_out[7] ) return true;
                            }
                        } else if( index_out[1]<tgt[2] && tgt[2]<this->n_y ){
                            if( 0<=tgt[1] && tgt[1]<this->n_z ){
                                if( index_out[3]<=tgt[0] && tgt[0]<=index_out[7] ) return true;
                            }
                        }
                    } else if( index_out[0]<tgt[3] && tgt[3]<index_out[4] ){
                        if( 0<=tgt[2] && tgt[2]<this->n_y ){
                            if( 0<=tgt[1] && tgt[1]<this->n_z ){
                                if( index_out[3]<=tgt[0] && tgt[0]<=index_out[7] ) return true;
                            }
                        }
                    } else if( tgt[3] == index_out[4] ){
                        if( tgt[2] == index_out[5] ){
                            if( 0<=tgt[1] && tgt[1]<=index_out[6] ){
                                if( index_out[3]<=tgt[0] && tgt[0]<=index_out[7] ) return true;
                            }
                        } else if( 0<=tgt[2] && tgt[2]<index_out[5] ){
                            if( 0<=tgt[1] && tgt[1]<this->n_z ){
                                if( index_out[3]<=tgt[0] && tgt[0]<=index_out[7] ) return true;
                            }
                        }
                    }
                }
                return false;
            }

            void _collect_buffer_info(){
                //--- collect range info
                _mpi::allgather(this->my_n_grid_in , this->n_grid_in_list );
                _mpi::allgather(this->my_n_grid_out, this->n_grid_out_list);
                _mpi::allgather(this->my_index_in  , this->index_in_list  );
                _mpi::allgather(this->my_index_out , this->index_out_list );
            }

            template <size_t Ndim>
            void _prepare_transpose(){
                static_assert(Ndim == 3 or Ndim == 4);
                this->_check_c2c_only();

                const int n_proc  = _mpi::get_n_proc();
                const int my_rank = _mpi::get_rank();

                const size_t n_grid_total = this->n_x
                                          * this->n_y
                                          * this->n_z
                                          * std::max(this->n_w, 1);  // n_w == 0 in c2c_3D
                const size_t n_reserve    = static_cast<size_t>( n_grid_total/(n_proc*n_proc) + 4 );

                this->transpose_out_in_send_index.resize( n_proc );
                this->transpose_out_in_recv_index.resize( n_proc );
                this->transpose_in_out_send_index.resize( n_proc );
                this->transpose_in_out_recv_index.resize( n_proc );

                std::vector< std::array<int, Ndim> > index_in_seq, index_out_seq;
                index_in_seq.reserve( n_reserve);
                index_out_seq.reserve(n_reserve);

                //--- search between local output <-> other input
                index_out_seq.resize( this->get_n_grid_out() );
                this->gen_output_index_sequence( index_out_seq.data() );
                const auto index_out_local = this->get_index_out();
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    index_in_seq.resize( this->get_n_grid_in(i_proc) );
                    this->gen_input_index_sequence( index_in_seq.data(), i_proc );
                    const auto index_in_other = this->get_index_in( i_proc );

                    //------ make a projection of local output -> other input
                    auto& send_out_in_index = this->transpose_out_in_send_index[i_proc];
                    send_out_in_index.clear();
                    send_out_in_index.reserve(n_reserve);
                    for(size_t ii=0; ii<index_out_seq.size(); ++ii){
                        const auto index_out = index_out_seq[ii];

                        //------ range check (pruning)
                        if( ! this->_is_input_range(index_out, index_in_other) ) continue;

                        for(size_t jj=0; jj<index_in_seq.size(); ++jj){
                            if(index_out == index_in_seq[jj]){
                                send_out_in_index.emplace_back(ii);
                                break;
                            }
                        }
                    }

                    //------ make a projection of local output <- other input
                    auto& recv_in_out_index = this->transpose_in_out_recv_index[i_proc];
                    recv_in_out_index.clear();
                    recv_in_out_index.reserve(n_reserve);
                    for(size_t ii=0; ii<index_in_seq.size(); ++ii){
                        const auto index_in = index_in_seq[ii];

                        //------ range check (pruning)
                        if( ! this->_is_output_range(index_in, index_out_local) ) continue;

                        for(size_t jj=0; jj<index_out_seq.size(); ++jj){
                            if(index_in == index_out_seq[jj]){
                                recv_in_out_index.emplace_back(jj);
                                break;
                            }
                        }
                    }
                }

                //--- search between local input <-> other output
                index_in_seq.resize( this->get_n_grid_in() );
                this->gen_input_index_sequence( index_in_seq.data() );
                const auto index_in_local = this->get_index_in();
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    index_out_seq.resize( this->get_n_grid_out(i_proc) );
                    this->gen_output_index_sequence( index_out_seq.data(), i_proc );
                    const auto index_out_other = this->get_index_out( i_proc );

                    //------ make a projection of local input -> other output
                    auto& send_in_out_index = this->transpose_in_out_send_index[i_proc];
                    send_in_out_index.clear();
                    send_in_out_index.reserve(n_reserve);
                    for(size_t ii=0; ii<index_in_seq.size(); ++ii){
                        const auto index_in = index_in_seq[ii];

                        //------ range check (pruning)
                        if( ! this->_is_output_range(index_in, index_out_other) ) continue;

                        for(size_t jj=0; jj<index_out_seq.size(); ++jj){
                            if(index_in == index_out_seq[jj]){
                                send_in_out_index.emplace_back(ii);
                                break;
                            }
                        }
                    }

                    //------ make a projection of local input <- other output
                    auto& recv_out_in_index = this->transpose_out_in_recv_index[i_proc];
                    recv_out_in_index.clear();
                    recv_out_in_index.reserve(n_reserve);
                    for(size_t ii=0; ii<index_out_seq.size(); ++ii){
                        const auto index_out = index_out_seq[ii];

                        //------ range check (pruning)
                        if( ! this->_is_input_range(index_out, index_in_local) ) continue;

                        for(size_t jj=0; jj<index_in_seq.size(); ++jj){
                            if(index_out == index_in_seq[jj]){
                                recv_out_in_index.emplace_back(jj);
                                break;
                            }
                        }
                    }
                }

                //--- check consistency
                this->_check_transpose_table();

                //--- check MPI communicate is neccesary or not in transpose arrays.
                bool mpi_in_out_flag = false;
                bool mpi_out_in_flag = false;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    if(i_proc == my_rank) continue;
                    if(this->transpose_in_out_send_index.at(i_proc).size() > 0) mpi_in_out_flag = true;
                    if(this->transpose_out_in_send_index.at(i_proc).size() > 0) mpi_out_in_flag = true;
                }
                this->transpose_in_out_mpi_flag = _mpi::sync_OR(mpi_in_out_flag);
                this->transpose_out_in_mpi_flag = _mpi::sync_OR(mpi_out_in_flag);

                //--- preparing for transpose was completed
                this->transpose_flag = true;
            }

            void _check_transpose_table() const {
                const int n_proc  = _mpi::get_n_proc();

                //--- check send grid size
                int n_grid_in  = 0;
                int n_grid_out = 0;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    n_grid_in  += this->transpose_in_out_send_index.at(i_proc).size();
                    n_grid_out += this->transpose_out_in_send_index.at(i_proc).size();
                }
                if( n_grid_in != this->get_n_grid_in() ){
                    std::ostringstream oss;
                    oss << "internal error: failure to make 'transpose_in_out_send_index'\n"
                        << "   total len = " << n_grid_in
                        << ", must be = " << this->get_n_grid_in() << " (= n_grid_in).\n";
                    throw std::logic_error(oss.str());
                }
                if( n_grid_out != this->get_n_grid_out() ){
                    std::ostringstream oss;
                    oss << "internal error: failure to make 'transpose_out_in_send_index'\n"
                        << "   total len = " << n_grid_out
                        << ", must be = " << this->get_n_grid_out() << " (= n_grid_out).\n";
                    throw std::logic_error(oss.str());
                }

                //--- check recv grid size
                n_grid_in  = 0;
                n_grid_out = 0;
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    n_grid_in  += this->transpose_out_in_recv_index.at(i_proc).size();
                    n_grid_out += this->transpose_in_out_recv_index.at(i_proc).size();
                }
                if( n_grid_in != this->get_n_grid_in() ){
                    std::ostringstream oss;
                    oss << "internal error: failure to make 'transpose_out_in_recv_index'\n"
                        << "   total len = " << n_grid_in
                        << ", must be = " << this->get_n_grid_in() << " (= n_grid_in).\n";
                    throw std::logic_error(oss.str());
                }
                if( n_grid_out != this->get_n_grid_out() ){
                    std::ostringstream oss;
                    oss << "internal error: failure to make 'transpose_in_out_recv_index'\n"
                        << "   total len = " << n_grid_out
                        << ", must be = " << this->get_n_grid_out() << " (= n_grid_out).\n";
                    throw std::logic_error(oss.str());
                }
            }
        };

    }

}
