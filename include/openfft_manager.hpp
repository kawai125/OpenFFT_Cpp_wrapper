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

        //----------------------------------------------------------------------
        //    FFT & IFFT wrapper
        //----------------------------------------------------------------------
        template <class Alloc_f, class Alloc_c>
        void fft_r2c_3d_forward(std::vector<float_t  , Alloc_f> &input,
                                std::vector<complex_t, Alloc_c> &output){

            //--- check buffer size
            this->_check_buffer_length(input, this->get_n_grid_in());
            output.resize(this->get_n_grid_out());

            //--- perform FFT
            this->fft_r2c_3d_forward(input.data(), output.data());
        }
        void fft_r2c_3d_forward(float_t   *input,
                                complex_t *output){
            _impl::fp64_global_mngr.fft_r2c_3d_forward(input, output);
        }

        template <class Alloc_L, class Alloc_R>
        void fft_c2c_3d_forward(std::vector<complex_t, Alloc_L> &input,
                                std::vector<complex_t, Alloc_R> &output){

            //--- check buffer size
            this->_check_buffer_length(input, this->get_n_grid_in());
            output.resize(this->get_n_grid_out());

            //--- perform FFT
            this->fft_c2c_3d_forward(input.data(), output.data());
        }
        void fft_c2c_3d_forward(complex_t *input,
                                complex_t *output){
            _impl::fp64_global_mngr.fft_c2c_3d_forward(input, output);
        }
        template <class Alloc_L, class Alloc_R>
        void fft_c2c_3d_backward(std::vector<complex_t, Alloc_L> &input,
                                 std::vector<complex_t, Alloc_R> &output){

            //--- check buffer size
            this->_check_buffer_length(input, this->get_n_grid_in());
            output.resize(this->get_n_grid_out());

            //--- perform FFT
            this->fft_c2c_3d_backward(input.data(), output.data());
        }
        void fft_c2c_3d_backward(complex_t *input,
                                 complex_t *output){
            _impl::fp64_global_mngr.fft_c2c_3d_backward(input, output);
        }


        template <class Alloc_L, class Alloc_R>
        void fft_c2c_4d_forward(std::vector<complex_t, Alloc_L> &input,
                                std::vector<complex_t, Alloc_R> &output){

            //--- check buffer size
            this->_check_buffer_length(input, this->get_n_grid_in());
            output.resize(this->get_n_grid_out());

            //--- perform FFT
            this->fft_c2c_4d_forward(input.data(), output.data());
        }
        void fft_c2c_4d_forward(complex_t *input,
                                complex_t *output){
            _impl::fp64_global_mngr.fft_c2c_4d_forward(input, output);
        }
        /*  under developping
        template <class Alloc_L, class Alloc_R>
        void fft_c2c_4d_backward(std::vector<complex_t, Alloc_L> &input,
                                 std::vector<complex_t, Alloc_R> &output){

            //--- check buffer size
            this->_check_buffer_length(input, this->get_n_grid_in());
            output.resize(this->get_n_grid_out());

            //--- perform FFT
            this->fft_c2c_4d_backward(input.data(), output.data());
        }
        void fft_c2c_4d_backward(complex_t *input,
                                 complex_t *output){
            _impl::fp64_global_mngr.fft_c2c_4d_backward(input, output);
        }
        */

        //----------------------------------------------------------------------
        //    input buffer manipulator (3D)
        //----------------------------------------------------------------------
        //--- at local process
        template <class T_3d ,
                  class T_buf, class Alloc_buf>
        void copy_3d_array_into_input_buffer(const T_3d                          *array_3d,
                                                   std::vector<T_buf, Alloc_buf> &buffer   ) const {
            buffer.resize(this->get_n_grid_in());
            this->copy_3d_array_into_input_buffer(array_3d, buffer.data());
        }
        template <class T_3d ,
                  class T_buf >
        void copy_3d_array_into_input_buffer(const T_3d  *array_3d,
                                                   T_buf *buffer   ) const {
            CopyIntoBuffer<T_3d, T_buf> copy_into_buffer;
            this->apply_3d_array_with_input_buffer(array_3d, buffer, copy_into_buffer);
        }
        template <class T_3d ,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_3d_array_with_input_buffer(T_3d                          *array_3d,
                                                   std::vector<T_buf, Alloc_buf> &buffer,
                                                   ApplyFunc                      func     ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_in());
            return this->apply_3d_array_with_input_buffer(array_3d, buffer.data(), func);
        }
        template <class T_3d ,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_3d_array_with_input_buffer(T_3d      *array_3d,
                                                   T_buf     *buffer,
                                                   ApplyFunc  func     ) const {
            return _impl::fp64_global_mngr.apply_3d_array_with_input_buffer(array_3d, buffer, func);
        }

        //--- for other process data part (if needed)
        template <class T_3d ,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_3d_array_with_input_buffer(      T_3d                          *array_3d,
                                                         std::vector<T_buf, Alloc_buf> &buffer,
                                                         ApplyFunc                      func,
                                                   const int                            i_proc) const {
            this->_check_buffer_length(buffer, this->get_n_grid_in());
            return this->apply_3d_array_with_input_buffer(array_3d, buffer.data(), func, i_proc);
        }
        template <class T_3d ,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_3d_array_with_input_buffer(      T_3d      *array_3d,
                                                         T_buf     *buffer,
                                                         ApplyFunc  func,
                                                   const int        i_proc   ) const {
            return _impl::fp64_global_mngr.apply_3d_array_with_input_buffer(array_3d, buffer, func, i_proc);
        }

        //----------------------------------------------------------------------
        //    input buffer manipulator (4D)
        //----------------------------------------------------------------------
        //--- at local process
        template <class T_4d ,
                  class T_buf, class Alloc_buf>
        void copy_4d_array_into_input_buffer(const T_4d                          *array_4d,
                                                   std::vector<T_buf, Alloc_buf> &buffer   ) const {
            buffer.resize(this->get_n_grid_in());
            this->copy_4d_array_into_input_buffer(array_4d, buffer.data());
        }
        template <class T_4d ,
                  class T_buf >
        void copy_4d_array_into_input_buffer(const T_4d  *array_4d,
                                                   T_buf *buffer   ) const {
            CopyIntoBuffer<T_4d, T_buf> copy_into_buffer;
            this->apply_4d_array_with_input_buffer(array_4d, buffer, copy_into_buffer);
        }
        template <class T_4d ,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_4d_array_with_input_buffer(T_4d                          *array_4d,
                                                   std::vector<T_buf, Alloc_buf> &buffer,
                                                   ApplyFunc                      func     ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_in());
            return this->apply_4d_array_with_input_buffer(array_4d, buffer.data(), func);
        }
        template <class T_4d ,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_4d_array_with_input_buffer(T_4d      *array_4d,
                                                   T_buf     *buffer,
                                                   ApplyFunc  func     ) const {
            return _impl::fp64_global_mngr.apply_4d_array_with_input_buffer(array_4d, buffer, func);
        }

        //--- for other process data part (if needed)
        template <class T_4d ,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_4d_array_with_input_buffer(      T_4d                          *array_4d,
                                                         std::vector<T_buf, Alloc_buf> &buffer,
                                                         ApplyFunc                      func,
                                                   const int                            i_proc   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_in());
            return this->apply_4d_array_with_input_buffer(array_4d, buffer.data(), func, i_proc);
        }
        template <class T_4d ,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_4d_array_with_input_buffer(      T_4d      *array_4d,
                                                         T_buf     *buffer,
                                                         ApplyFunc  func,
                                                   const int        i_proc   ) const {
            return _impl::fp64_global_mngr.apply_4d_array_with_input_buffer(array_4d, buffer, func, i_proc);
        }

        //----------------------------------------------------------------------
        //    output buffer manipulator (3D)
        //----------------------------------------------------------------------
        //--- at local process
        template <class T_3d ,
                  class T_buf, class Alloc_buf>
        void copy_3d_array_from_output_buffer(      T_3d                          *array_3d,
                                              const std::vector<T_buf, Alloc_buf> &buffer   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out());
            this->copy_3d_array_from_output_buffer(array_3d, buffer.data());
        }
        template <class T_3d ,
                  class T_buf >
        void copy_3d_array_from_output_buffer(      T_3d  *array_3d,
                                              const T_buf *buffer   ) const {
            CopyFromBuffer<T_3d, T_buf> copy_from_buffer;
            this->apply_3d_array_with_output_buffer(array_3d, buffer, copy_from_buffer);
        }
        template <class T_3d ,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_3d_array_with_output_buffer(T_3d                          *array_3d,
                                                    std::vector<T_buf, Alloc_buf> &buffer,
                                                    ApplyFunc                      func     ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out());
            return this->apply_3d_array_with_output_buffer(array_3d, buffer.data(), func);
        }
        template <class T_3d ,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_3d_array_with_output_buffer(T_3d      *array_3d,
                                                    T_buf     *buffer,
                                                    ApplyFunc  func     ) const {
            return _impl::fp64_global_mngr.apply_3d_array_with_output_buffer(array_3d, buffer, func);
        }

        //--- for other process data part (if needed)
        template <class T_3d ,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_3d_array_with_output_buffer(      T_3d                          *array_3d,
                                                          std::vector<T_buf, Alloc_buf> &buffer,
                                                          ApplyFunc                      func,
                                                    const int                            i_proc   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out());
            return this->apply_3d_array_with_output_buffer(array_3d, buffer.data(), func, i_proc);
        }
        template <class T_3d ,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_3d_array_with_output_buffer(      T_3d      *array_3d,
                                                          T_buf     *buffer,
                                                          ApplyFunc  func,
                                                    const int        i_proc) const {
            return _impl::fp64_global_mngr.apply_3d_array_with_output_buffer(array_3d, buffer, func, i_proc);
        }

        //----------------------------------------------------------------------
        //    output buffer manipulator (4D)
        //----------------------------------------------------------------------
        //--- at local process
        template <class T_4d ,
                  class T_buf, class Alloc_buf>
        void copy_4d_array_from_output_buffer(      T_4d                          *array_4d,
                                              const std::vector<T_buf, Alloc_buf> &buffer   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out());
            this->copy_4d_array_from_output_buffer(array_4d, buffer.data());
        }
        template <class T_4d ,
                  class T_buf >
        void copy_4d_array_from_output_buffer(      T_4d  *array_4d,
                                              const T_buf *buffer   ) const {
            CopyFromBuffer<T_4d, T_buf> copy_from_buffer;
            this->apply_4d_array_with_output_buffer(array_4d, buffer, copy_from_buffer);
        }
        template <class T_4d ,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_4d_array_with_output_buffer(T_4d                          *array_4d,
                                                    std::vector<T_buf, Alloc_buf> &buffer,
                                                    ApplyFunc                      func     ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out());
            return this->apply_4d_array_with_output_buffer(array_4d, buffer.data(), func);
        }
        template <class T_4d ,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_4d_array_with_output_buffer(T_4d      *array_4d,
                                                    T_buf     *buffer,
                                                    ApplyFunc  func     ) const {
            return _impl::fp64_global_mngr.apply_4d_array_with_output_buffer(array_4d, buffer, func);
        }

        //--- for other process data part (if needed)
        template <class T_4d ,
                  class T_buf, class Alloc_buf,
                  class ApplyFunc >
        ApplyFunc apply_4d_array_with_output_buffer(      T_4d                          *array_4d,
                                                          std::vector<T_buf, Alloc_buf> &buffer,
                                                          ApplyFunc                      func,
                                                    const int                            i_proc   ) const {
            this->_check_buffer_length(buffer, this->get_n_grid_out());
            return this->apply_4d_array_with_output_buffer(array_4d, buffer.data(), func, i_proc);
        }
        template <class T_4d ,
                  class T_buf,
                  class ApplyFunc >
        ApplyFunc apply_4d_array_with_output_buffer(      T_4d      *array_4d,
                                                          T_buf     *buffer,
                                                          ApplyFunc  func,
                                                    const int        i_proc   ) const {
            return _impl::fp64_global_mngr.apply_4d_array_with_output_buffer(array_4d, buffer, func, i_proc);
        }

        //----------------------------------------------------------------------
        //    output_buffer to input_buffer converter
        //----------------------------------------------------------------------
        template <class Alloc_L, class Alloc_R>
        void convert_output_to_input(      std::vector<complex_t, Alloc_L> &input_buf,
                                     const std::vector<complex_t, Alloc_R> &output_buf){

            //--- check buffer size
            this->_check_buffer_length( output_buf, this->get_n_grid_out() );
            input_buf.resize( this->get_n_grid_in() );

            this->convert_output_to_input(input_buf.data(), output_buf.data());
        }
        void convert_output_to_input(      complex_t *input_buf,
                                     const complex_t *output_buf){
            _impl::fp64_global_mngr.convert_output_to_input(input_buf, output_buf);
        }

        //----------------------------------------------------------------------
        //    gather inferface for global 3D/4D array from output_buffer
        //----------------------------------------------------------------------
        template <class Alloc>
        void gather_3d_array(      complex_t                     *array_3d,
                             const std::vector<complex_t, Alloc> &output_buf,
                             const int                            tgt_proc   ){
            this->_check_buffer_length(output_buf, this->get_n_grid_out());
            this->gather_3d_array(array_3d, output_buf.data(), tgt_proc);
        }
        void gather_3d_array(      complex_t *array_3d,
                             const complex_t *output_buf,
                             const int        tgt_proc   ){
            _impl::fp64_global_mngr.gather_3d_array(array_3d, output_buf, tgt_proc);
        }
        template <class Alloc>
        void allgather_3d_array(      complex_t                     *array_3d,
                                const std::vector<complex_t, Alloc> &output_buf ){
            this->_check_buffer_length(output_buf, this->get_n_grid_out());
            this->allgather_3d_array(array_3d, output_buf.data());
        }
        void allgather_3d_array(      complex_t *array_3d,
                                const complex_t *output_buf){
            _impl::fp64_global_mngr.allgather_3d_array(array_3d, output_buf);
        }

        template <class Alloc>
        void gather_4d_array(      complex_t                     *array_4d,
                             const std::vector<complex_t, Alloc> &output_buf,
                             const int                            tgt_proc   ){
            this->_check_buffer_length(output_buf, this->get_n_grid_out());
            this->gather_4d_array(array_4d, output_buf.data(), tgt_proc);
        }
        void gather_4d_array(      complex_t *array_4d,
                             const complex_t *output_buf,
                             const int        tgt_proc   ){
            _impl::fp64_global_mngr.gather_4d_array(array_4d, output_buf, tgt_proc);
        }
        template <class Alloc>
        void allgather_4d_array(      complex_t                     *array_4d,
                                const std::vector<complex_t, Alloc> &output_buf ){
            this->_check_buffer_length(output_buf, this->get_n_grid_out());
            this->allgather_4d_array(array_4d, output_buf.data());
        }
        void allgather_4d_array(      complex_t *array_4d,
                                const complex_t *output_buf){
            _impl::fp64_global_mngr.allgather_4d_array(array_4d, output_buf);
        }

        //----------------------------------------------------------------------
        //    internal table info (for debug)
        //----------------------------------------------------------------------
        void report_convert_matrix(){ _impl::fp64_global_mngr.report_convert_matrix(); }

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
    };

}
