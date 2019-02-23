/**************************************************************************************************/
/**
* @file  openfft_manager.hpp
* @brief wrapper for OpenFFT library.
*
*   OpenFFT library
*   http://www.openmx-square.org/openfft/
*/
/**************************************************************************************************/
#pragma once

#include <array>
#include <vector>
#include <sstream>
#include <exception>
#include <cstdint>

#include "openfft_defs.hpp"
#include "openfft_global_manager.hpp"


namespace OpenFFT {

    namespace _impl {
        static GlobalManager<double> fp64_global_mngr;
    }

    template <class Tfloat>
    class Manager;

    /*
    *  @brief wrapper interface for OpenFFT
    */
    template <>
    class Manager<double>{
    private:
        using float_t   = double;
        using complex_t = typename _impl::GlobalManager<double>::complex_type;

    public:
        using float_type   = float_t;
        using complex_type = complex_t;
        using IndexList    = typename _impl::GlobalManager<double>::IndexList;

        Manager() = default;
        ~Manager() = default;

        //--- copy is prohibited
        template <class Trhs>
        Manager(const Manager<Trhs> &rv) = delete;
        template <class Trhs>
        Manager& operator = (const Manager<Trhs> &rv) = delete;

        //--- initializer
        void init_r2c_3d(const int n_z, const int n_y, const int n_x,
                         const int offt_measure,
                         const int measure_time,
                         const int print_memory ){

            if( _impl::fp64_global_mngr.is_same_grid(n_z, n_y, n_x, FFT_GridType::r2c_3D) ) return;

            _impl::fp64_global_mngr.init_r2c_3d(n_z, n_y, n_x,
                                                offt_measure,
                                                measure_time,
                                                print_memory  );
        }
        void init_r2c_3d(const int n_z, const int n_y, const int n_x){
            const int offt_measure = 0;  // auto-tuning
            const int measure_time = 0;  // not use
            const int print_memory = 0;  // not use
            this->init_r2c_3d(n_z, n_y, n_x,
                              offt_measure,
                              measure_time,
                              print_memory );
        }
        void init_c2c_3d(const int n_z, const int n_y, const int n_x,
                         const int offt_measure,
                         const int measure_time,
                         const int print_memory ){

            if( _impl::fp64_global_mngr.is_same_grid(n_z, n_y, n_x, FFT_GridType::c2c_3D) ) return;

            _impl::fp64_global_mngr.init_c2c_3d(n_z, n_y, n_x,
                                                offt_measure,
                                                measure_time,
                                                print_memory  );
        }
        void init_c2c_3d(const int n_z, const int n_y, const int n_x){
            const int offt_measure = 0;  // auto-tuning
            const int measure_time = 0;  // not use
            const int print_memory = 0;  // not use
            this->init_c2c_3d(n_z, n_y, n_x,
                              offt_measure,
                              measure_time,
                              print_memory );
        }
        void init_c2c_4d(const int n_w, const int n_z, const int n_y, const int n_x,
                         const int offt_measure,
                         const int measure_time,
                         const int print_memory ){

            if( _impl::fp64_global_mngr.is_same_grid(n_w, n_z, n_y, n_x, FFT_GridType::c2c_4D) ) return;

            _impl::fp64_global_mngr.init_c2c_4d(n_w, n_z, n_y, n_x,
                                                offt_measure,
                                                measure_time,
                                                print_memory  );
        }
        void init_c2c_4d(const int n_w, const int n_z, const int n_y, const int n_x){
            const int offt_measure = 0;  // auto-tuning
            const int measure_time = 0;  // not use
            const int print_memory = 0;  // not use
            this->init_c2c_4d(n_w, n_z, n_y, n_x,
                              offt_measure,
                              measure_time,
                              print_memory );
        }
        void finalize(){
            _impl::fp64_global_mngr.finalize();
        }

        //--- get local proc info
        inline int       get_max_n_grid() const { return _impl::fp64_global_mngr.get_max_n_grid(); }
        inline int       get_n_grid_in()  const { return _impl::fp64_global_mngr.get_n_grid_in();  }
        inline int       get_n_grid_out() const { return _impl::fp64_global_mngr.get_n_grid_out(); }
        inline IndexList get_index_in()   const { return _impl::fp64_global_mngr.get_index_in();   }
        inline IndexList get_index_out()  const { return _impl::fp64_global_mngr.get_index_out();  }

        //--- get other proc info
        inline int       get_n_grid_in( const int i_proc) const { return _impl::fp64_global_mngr.get_n_grid_in( i_proc); }
        inline int       get_n_grid_out(const int i_proc) const { return _impl::fp64_global_mngr.get_n_grid_out(i_proc); }
        inline IndexList get_index_in(  const int i_proc) const { return _impl::fp64_global_mngr.get_index_in(  i_proc); }
        inline IndexList get_index_out( const int i_proc) const { return _impl::fp64_global_mngr.get_index_out( i_proc); }

        //--- elapsed time
        inline double get_time() const { return _impl::fp64_global_mngr.get_time(); }

        //--- grid info
        FFT_GridType get_grid_type() const { return _impl::fp64_global_mngr.get_grid_type(); }

        void get_grid_size(int &nz, int &ny, int &nx) const {
            _impl::fp64_global_mngr.get_grid_size(nz, ny, nx);
        }
        void get_grid_size(int &nw, int &nz, int &ny, int &nx) const {
            _impl::fp64_global_mngr.get_grid_size(nw, nz, ny, nx);
        }

        //----------------------------------------------------------------------
        //    FFT & IFFT wrapper
        //----------------------------------------------------------------------
        template <class Alloc_f, class Alloc_c>
        void fft_r2c_forward(std::vector<float_t  , Alloc_f> &input,
                             std::vector<complex_t, Alloc_c> &output){

            //--- check buffer size
            this->_reserve_buffer_capacity(input);
            this->_reserve_buffer_capacity(output);
            //--- check data length
            this->_check_buffer_length(input, this->get_n_grid_in());
            output.resize(this->get_n_grid_out());

            //--- perform FFT
            this->fft_r2c_forward(input.data(), output.data());
        }
        void fft_r2c_forward(float_t   *input,
                             complex_t *output){
            _impl::fp64_global_mngr.fft_r2c_forward(input, output);
        }

        template <class Alloc_L, class Alloc_R>
        void fft_c2c_forward(std::vector<complex_t, Alloc_L> &input,
                             std::vector<complex_t, Alloc_R> &output){

            //--- check buffer size
            this->_reserve_buffer_capacity(input);
            this->_reserve_buffer_capacity(output);
            //--- check data length
            this->_check_buffer_length(input, this->get_n_grid_in());
            output.resize(this->get_n_grid_out());

            //--- perform FFT
            this->fft_c2c_forward(input.data(), output.data());
        }
        void fft_c2c_forward(complex_t *input,
                             complex_t *output){
            _impl::fp64_global_mngr.fft_c2c_forward(input, output);
        }
        template <class Alloc_L, class Alloc_R>
        void fft_c2c_backward(std::vector<complex_t, Alloc_L> &input,
                              std::vector<complex_t, Alloc_R> &output){

            //--- check buffer size
            this->_reserve_buffer_capacity(input);
            this->_reserve_buffer_capacity(output);
            //--- check data length
            this->_check_buffer_length(input, this->get_n_grid_in());
            output.resize(this->get_n_grid_out());

            //--- perform FFT
            this->fft_c2c_backward(input.data(), output.data());
        }
        void fft_c2c_backward(complex_t *input,
                              complex_t *output){
            _impl::fp64_global_mngr.fft_c2c_backward(input, output);
        }

        //----------------------------------------------------------------------
        //    input buffer manipulator
        //----------------------------------------------------------------------
        //--- at local process
        template <class T_arr,
                  class T_buf, class Alloc_buf>
        void copy_array_into_input_buffer(const T_arr                         *array,
                                                std::vector<T_buf, Alloc_buf> &buffer) const {
            buffer.resize(this->get_n_grid_in());
            this->copy_array_into_input_buffer(array, buffer.data());
        }
        template <class T_arr,
                  class T_buf >
        void copy_array_into_input_buffer(const T_arr *array,
                                                T_buf *buffer) const {
            this->apply_array_with_input_buffer(array, buffer, CopyIntoBuffer{});
        }

        template <class T_arr,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_input_buffer(      T_arr                         *array,
                                                const std::vector<T_buf, Alloc_buf> &buffer,
                                                      ApplyFunc                      func   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_in());
            const T_buf* buf_ptr = buffer.data();
            return this->apply_array_with_input_buffer(array, buf_ptr, func);
        }
        template <class T_arr,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_input_buffer(T_arr                         *array,
                                                std::vector<T_buf, Alloc_buf> &buffer,
                                                ApplyFunc                      func   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_in());
            return this->apply_array_with_input_buffer(array, buffer.data(), func);
        }
        template <class T_arr,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_input_buffer(T_arr     *array,
                                                T_buf     *buffer,
                                                ApplyFunc  func   ) const {
            return _impl::fp64_global_mngr.apply_array_with_input_buffer(array, buffer, func);
        }

        //--- for other process data part (if needed)
        template <class T_arr,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_input_buffer(      T_arr                         *array,
                                                const std::vector<T_buf, Alloc_buf> &buffer,
                                                      ApplyFunc                      func,
                                                const int                            i_proc) const {
            this->_check_buffer_length(buffer, this->get_n_grid_in(i_proc));
            const T_buf* buf_ptr = buffer.data();
            return this->apply_array_with_input_buffer(array, buf_ptr, func, i_proc);
        }
        template <class T_arr,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_input_buffer(      T_arr                         *array,
                                                      std::vector<T_buf, Alloc_buf> &buffer,
                                                      ApplyFunc                      func,
                                                const int                            i_proc ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_in(i_proc));
            return this->apply_array_with_input_buffer(array, buffer.data(), func, i_proc);
        }
        template <class T_arr,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_input_buffer(      T_arr     *array,
                                                      T_buf     *buffer,
                                                      ApplyFunc  func,
                                                const int        i_proc ) const {
            return _impl::fp64_global_mngr.apply_array_with_input_buffer(array, buffer, func, i_proc);
        }

        //----------------------------------------------------------------------
        //    output buffer manipulator
        //----------------------------------------------------------------------
        //--- at local process
        template <class T_arr,
                  class T_buf, class Alloc_buf>
        void copy_array_from_output_buffer(      T_arr                         *array,
                                           const std::vector<T_buf, Alloc_buf> &buffer) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out());
            const T_buf* buf_ptr = buffer.data();
            this->copy_array_from_output_buffer(array, buf_ptr);
        }
        template <class T_arr,
                  class T_buf >
        void copy_array_from_output_buffer(      T_arr *array,
                                           const T_buf *buffer) const {
            this->apply_array_with_output_buffer(array, buffer, CopyFromBuffer{});
        }

        template <class T_arr,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_output_buffer(      T_arr                         *array,
                                                 const std::vector<T_buf, Alloc_buf> &buffer,
                                                       ApplyFunc                      func   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out());
            const T_buf* buf_ptr = buffer.data();
            return this->apply_array_with_output_buffer(array, buf_ptr, func);
        }
        template <class T_arr,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_output_buffer(T_arr                         *array,
                                                 std::vector<T_buf, Alloc_buf> &buffer,
                                                 ApplyFunc                      func   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out());
            return this->apply_array_with_output_buffer(array, buffer.data(), func);
        }
        template <class T_arr,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_output_buffer(T_arr     *array,
                                                 T_buf     *buffer,
                                                 ApplyFunc  func     ) const {
            return _impl::fp64_global_mngr.apply_array_with_output_buffer(array, buffer, func);
        }

        //--- for other process data part (if needed)
        template <class T_arr,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_output_buffer(      T_arr                         *array,
                                                 const std::vector<T_buf, Alloc_buf> &buffer,
                                                       ApplyFunc                      func,
                                                 const int                            i_proc   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out(i_proc));
            const T_buf* buf_ptr = buffer.data();
            return this->apply_array_with_output_buffer(array, buf_ptr, func, i_proc);
        }
        template <class T_arr,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_output_buffer(      T_arr                         *array,
                                                       std::vector<T_buf, Alloc_buf> &buffer,
                                                       ApplyFunc                      func,
                                                 const int                            i_proc ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out(i_proc));
            return this->apply_array_with_output_buffer(array, buffer.data(), func, i_proc);
        }
        template <class T_arr,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_array_with_output_buffer(      T_arr     *array,
                                                       T_buf     *buffer,
                                                       ApplyFunc  func,
                                                 const int        i_proc) const {
            return _impl::fp64_global_mngr.apply_array_with_output_buffer(array, buffer, func, i_proc);
        }

        //----------------------------------------------------------------------
        //    index sequence generator
        //----------------------------------------------------------------------
        template <class Alloc_seq>
        void gen_input_index_sequence(std::vector<std::array<int, 3>, Alloc_seq> &index_seq) const {
            index_seq.resize( this->get_n_grid_in() );
            _impl::fp64_global_mngr.gen_input_index_sequence( index_seq.data() );
        }
        template <class Alloc_seq>
        void gen_input_index_sequence(      std::vector<std::array<int, 3>, Alloc_seq> &index_seq,
                                      const int                                         i_proc    ) const {
            index_seq.resize( this->get_n_grid_in(i_proc) );
            _impl::fp64_global_mngr.gen_input_index_sequence(index_seq.data(), i_proc);
        }
        template <class Alloc_seq>
        void gen_output_index_sequence(std::vector<std::array<int, 3>, Alloc_seq> &index_seq) const {
            index_seq.resize( this->get_n_grid_out() );
            _impl::fp64_global_mngr.gen_output_index_sequence( index_seq.data() );
        }
        template <class Alloc_seq>
        void gen_output_index_sequence(      std::vector<std::array<int, 3>, Alloc_seq> &index_seq,
                                       const int                                         i_proc    ) const {
            index_seq.resize( this->get_n_grid_out(i_proc) );
            _impl::fp64_global_mngr.gen_output_index_sequence(index_seq.data(), i_proc);
        }

        template <class Alloc_seq>
        void gen_input_index_sequence(std::vector<std::array<int, 4>, Alloc_seq> &index_seq) const {
            index_seq.resize( this->get_n_grid_in() );
            _impl::fp64_global_mngr.gen_input_index_sequence( index_seq.data() );
        }
        template <class Alloc_seq>
        void gen_input_index_sequence(      std::vector<std::array<int, 4>, Alloc_seq> &index_seq,
                                      const int                                         i_proc    ) const {
            index_seq.resize( this->get_n_grid_in(i_proc) );
            _impl::fp64_global_mngr.gen_input_index_sequence(index_seq.data(), i_proc);
        }
        template <class Alloc_seq>
        void gen_output_index_sequence(std::vector<std::array<int, 4>, Alloc_seq> &index_seq) const {
            index_seq.resize( this->get_n_grid_out() );
            _impl::fp64_global_mngr.gen_output_index_sequence( index_seq.data() );
        }
        template <class Alloc_seq>
        void gen_output_index_sequence(      std::vector<std::array<int, 4>, Alloc_seq> &index_seq,
                                       const int                                         i_proc    ) const {
            index_seq.resize( this->get_n_grid_out(i_proc) );
            _impl::fp64_global_mngr.gen_output_index_sequence(index_seq.data(), i_proc);
        }

        //----------------------------------------------------------------------
        //    transposer between output_buffer and input_buffer
        //----------------------------------------------------------------------
        template <class Tdata, class Alloc_L, class Alloc_R>
        void transpose_input_to_output(const std::vector<Tdata, Alloc_L> &input_buf,
                                             std::vector<Tdata, Alloc_R> &output_buf){

            //--- check buffer size
            this->_check_buffer_length( input_buf, this->get_n_grid_in() );
            output_buf.resize( this->get_n_grid_out() );

            this->transpose_input_to_output(input_buf.data(), output_buf.data());
        }
        template <class Tdata>
        void transpose_input_to_output(const Tdata *input_buf,
                                             Tdata *output_buf){
            _impl::fp64_global_mngr.transpose_input_to_output(input_buf, output_buf);
        }
        template <class Tdata, class Alloc_L, class Alloc_R>
        void transpose_output_to_input(const std::vector<Tdata, Alloc_L> &output_buf,
                                             std::vector<Tdata, Alloc_R> &input_buf ){

            //--- check buffer size
            this->_check_buffer_length( output_buf, this->get_n_grid_out() );
            input_buf.resize( this->get_n_grid_in() );

            this->transpose_output_to_input(output_buf.data(), input_buf.data());
        }
        template <class Tdata>
        void transpose_output_to_input(const Tdata *output_buf,
                                             Tdata *input_buf ){
            _impl::fp64_global_mngr.transpose_output_to_input(output_buf, input_buf);
        }

        //----------------------------------------------------------------------
        //    gather inferface for global 3D/4D array from output_buffer
        //----------------------------------------------------------------------
        template <class Tdata, class Alloc>
        void gather_array(      Tdata                     *array,
                          const std::vector<Tdata, Alloc> &output_buf,
                          const int                        tgt_proc   ){
            this->_check_buffer_length(output_buf, this->get_n_grid_out());
            this->gather_array(array, output_buf.data(), tgt_proc);
        }
        template <class Tdata>
        void gather_array(      Tdata *array,
                          const Tdata *output_buf,
                          const int    tgt_proc   ){
            _impl::fp64_global_mngr.gather_array(array, output_buf, tgt_proc);
        }
        template <class Tdata, class Alloc>
        void allgather_array(      Tdata                     *array,
                             const std::vector<Tdata, Alloc> &output_buf ){
            this->_check_buffer_length(output_buf, this->get_n_grid_out());
            this->allgather_array(array, output_buf.data());
        }
        template <class Tdata>
        void allgather_array(      Tdata *array,
                             const Tdata *output_buf){
            _impl::fp64_global_mngr.allgather_array(array, output_buf);
        }

    private:
        template <class Tf, class Alloc>
        void _check_buffer_length(const std::vector<Tf, Alloc> &buf, const size_t n_grid) const {
            if(buf.size() != n_grid){
                std::ostringstream oss;
                oss << "invalid buffer size." << "\n"
                    << "   buffer size = " << buf.size()
                    << ", must be = " << n_grid << "\n";
                throw std::length_error(oss.str());
            }
        }
        template <class Tf, class Alloc>
        void _check_buffer_capacity(const std::vector<Tf, Alloc> &buf) const {
            if( static_cast<int_fast64_t>(buf.capacity()) < this->get_max_n_grid() ){
                std::ostringstream oss;
                oss << "buffer capacity is not enough." << "\n"
                    << "   buffer capacity = " << buf.capacity()
                    << ", must be >= " << this->get_max_n_grid() << "\n";
                throw std::length_error(oss.str());
            }
        }
        template <class Tf, class Alloc>
        void _reserve_buffer_capacity(std::vector<Tf, Alloc> &buf) const {
            if( static_cast<int_fast64_t>(buf.capacity()) < this->get_max_n_grid() ){
                buf.reserve( this->get_max_n_grid() );
            }
        }
    };

}
