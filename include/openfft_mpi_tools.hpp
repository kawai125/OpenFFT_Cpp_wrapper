/**************************************************************************************************/
/**
* @file  openfft_mpi_tools.hpp
* @brief MPI wrapper for openfft_manager.hpp (local use)
*/
/**************************************************************************************************/
#pragma once

#include <vector>
#include <cassert>

#include <mpi.h>


namespace OpenFFT {
    namespace _impl {
        namespace _mpi {

            inline int get_n_proc(MPI_Comm comm = MPI_COMM_WORLD){
                int n_proc;
                MPI_Comm_size(comm, &n_proc);
                return n_proc;
            }
            inline int get_rank(MPI_Comm comm = MPI_COMM_WORLD){
                int rank;
                MPI_Comm_rank(comm, &rank);
                return rank;
            }

            inline void barrier(MPI_Comm comm = MPI_COMM_WORLD){
                MPI_Barrier(comm);
            }

            inline bool sync_OR(const bool flag, MPI_Comm comm = MPI_COMM_WORLD){
                bool global;
                bool local = flag;
                MPI_Allreduce(&local, &global, 1, MPI_C_BOOL, MPI_LOR, comm);
                return global;
            }


            template <class T>
            inline MPI_Datatype MakeDataSize(){
              static MPI_Datatype type = MPI_DATATYPE_NULL;
              if( type == MPI_DATATYPE_NULL ){
                  MPI_Type_contiguous(sizeof(T), MPI_BYTE, &type);
                  MPI_Type_commit(&type);
              }
              return type;
            };


            template <class Tdata, class Allocator, class Tindex>
            void serialize_vector_vector(const std::vector<std::vector<Tdata, Allocator>> &vec_vec,
                                               std::vector<Tdata, Allocator>              &vec_data,
                                               std::vector<Tindex>                        &vec_index){

                vec_data.clear();
                vec_index.clear();

                size_t len = 0;
                for(const auto& vec : vec_vec){
                    len += vec.size();
                }

                vec_data.reserve(len);
                vec_index.reserve(vec_vec.size()+1);

                size_t count = 0;
                for(auto &vec : vec_vec){
                    vec_index.emplace_back(count);  //start point
                    for(auto &elem : vec){
                        vec_data.emplace_back(elem);
                        ++count;
                    }
                }
                vec_index.push_back(count);  // terminater
            }
            template <class Tdata, class Allocator, class Tindex>
            void deserialize_vector_vector(const std::vector<Tdata, Allocator>              &vec_data,
                                           const std::vector<Tindex>                        &vec_index,
                                                 std::vector<std::vector<Tdata, Allocator>> &vec_vec   ){

                vec_vec.resize(vec_index.size()-1);

                for(size_t i=0; i<vec_index.size()-1; ++i){
                    vec_vec.at(i).clear();
                    size_t index_begin = vec_index.at(i);
                    size_t index_end   = vec_index.at(i+1);
                    for(size_t j=index_begin; j<index_end; ++j){
                        vec_vec.at(i).emplace_back(vec_data.at(j));
                    }
                }
            }

            template <class T, class Allocator>
            void gather(const T                         &value,
                              std::vector<T, Allocator> &recv_vec,
                        const int                        root,
                              MPI_Comm                   comm = MPI_COMM_WORLD){

                recv_vec.clear();
                recv_vec.resize(get_n_proc());

                MPI_Gather(&value,       1, MakeDataSize<T>(),
                           &recv_vec[0], 1, MakeDataSize<T>(),
                            root,
                            comm);
            }
            template <class T, class Allocator>
            void gather(const std::vector<T, Allocator>              &send_vec,
                              std::vector<std::vector<T, Allocator>> &recv_vec_vec,
                        const int                                     root,
                              MPI_Comm                                comm = MPI_COMM_WORLD){

                std::vector<int>          n_recv;
                std::vector<int>          n_recv_disp;
                std::vector<T, Allocator> recv_data;
                int                       len = send_vec.size();
                gather(len, n_recv, root, comm);

                const int n_proc = get_n_proc();
                n_recv_disp.resize(n_proc+1);
                n_recv_disp[0] = 0;
                for(int i=0; i<n_proc; ++i){
                    n_recv_disp.at(i+1) = n_recv_disp.at(i) + n_recv.at(i);
                }
                recv_data.resize( n_recv_disp[n_proc] );


                MPI_Gatherv(&send_vec[0],  send_vec.size(),             MakeDataSize<T>(),
                            &recv_data[0], &n_recv[0], &n_recv_disp[0], MakeDataSize<T>(),
                             root,
                             comm);

                if(get_rank() == root){
                    recv_vec_vec.resize(n_proc);
                    for(int i_proc=0; i_proc<n_proc; ++i_proc){
                        auto& local_vec = recv_vec_vec.at(i_proc);
                              local_vec.clear();
                              local_vec.reserve(n_recv.at(i_proc));

                        const int index_begin = n_recv_disp.at(i_proc);
                        const int index_end   = index_begin + n_recv.at(i_proc);
                        for(int index=index_begin; index<index_end; ++index){
                            local_vec.emplace_back( recv_data.at(index) );
                        }
                    }
                }
            }

            template <class T, class Allocator>
            void allgather(const T                         &value,
                                 std::vector<T, Allocator> &recv_vec,
                                 MPI_Comm                   comm = MPI_COMM_WORLD){

                recv_vec.clear();
                recv_vec.resize(get_n_proc());

                MPI_Allgather(&value      , 1, MakeDataSize<T>(),
                              &recv_vec[0], 1, MakeDataSize<T>(),
                               comm);
            }
            template <class T, class Allocator>
            void allgather(const std::vector<T, Allocator>              &send_vec,
                                 std::vector<std::vector<T, Allocator>> &recv_vec_vec,
                                 MPI_Comm                                comm = MPI_COMM_WORLD){

                const int n_proc = get_n_proc();

                std::vector<int>          n_recv;
                std::vector<int>          n_recv_disp;
                std::vector<T, Allocator> recv_data;
                int                      len = send_vec.size();
                allgather(len, n_recv, comm);

                n_recv_disp.resize(n_proc+1);
                n_recv_disp[0] = 0;
                for(int i=0; i<n_proc; ++i){
                    n_recv_disp.at(i+1) = n_recv_disp.at(i) + n_recv.at(i);
                }
                recv_data.resize( n_recv_disp[n_proc] );

                MPI_Allgatherv(&send_vec[0] , send_vec.size(),             MakeDataSize<T>(),
                               &recv_data[0], &n_recv[0], &n_recv_disp[0], MakeDataSize<T>(),
                                comm);

                recv_vec_vec.resize(n_proc);
                for(int i_proc=0; i_proc<n_proc; ++i_proc){
                    auto& local_vec = recv_vec_vec.at(i_proc);
                          local_vec.clear();
                          local_vec.reserve(n_recv.at(i_proc));

                    const int index_begin = n_recv_disp.at(i_proc);
                    const int index_end   = index_begin + n_recv.at(i_proc);
                    for(int index=index_begin; index<index_end; ++index){
                        local_vec.emplace_back( recv_data.at(index) );
                    }
                }
            }
            template <class T, class Allocator>
            void alltoall(const std::vector<T, Allocator> &send_vec,
                                std::vector<T, Allocator> &recv_vec,
                                MPI_Comm                   comm = MPI_COMM_WORLD){

                const int n_proc = get_n_proc();
                assert(send_vec.size() >= static_cast<size_t>(n_proc));

                recv_vec.resize(n_proc);

                MPI_Alltoall(&send_vec[0], 1, MakeDataSize<T>(),
                             &recv_vec[0], 1, MakeDataSize<T>(),
                              comm);
            }
            template <class T, class Allocator>
            void alltoall(const std::vector<std::vector<T, Allocator>> &send_vec_vec,
                                std::vector<std::vector<T, Allocator>> &recv_vec_vec,
                                MPI_Comm                                comm = MPI_COMM_WORLD){

                const int n_proc = get_n_proc();
                assert( send_vec_vec.size() >= static_cast<size_t>(n_proc) );

                std::vector<int> n_send;
                std::vector<int> n_send_disp;
                std::vector<int> n_recv;
                std::vector<int> n_recv_disp;

                n_send.resize(n_proc);
                n_recv.resize(n_proc);
                n_send_disp.resize(n_proc+1);
                n_recv_disp.resize(n_proc+1);

                n_send_disp[0] = 0;
                for(int i=0; i<n_proc; ++i){
                    n_send[i] = send_vec_vec[i].size();
                }
                alltoall(n_send, n_recv, comm);

                n_recv_disp[0] = 0;
                for(int i=0; i<n_proc; ++i){
                    n_recv_disp[i+1] = n_recv_disp[i] + n_recv[i];
                }

                std::vector<T, Allocator> send_data;
                std::vector<T, Allocator> recv_data;

                serialize_vector_vector(send_vec_vec, send_data, n_send_disp);
                recv_data.resize(n_recv_disp.back());

                MPI_Alltoallv(&send_data[0], &n_send[0], &n_send_disp[0], MakeDataSize<T>(),
                              &recv_data[0], &n_recv[0], &n_recv_disp[0], MakeDataSize<T>(),
                               comm);

                deserialize_vector_vector(recv_data, n_recv_disp, recv_vec_vec);
            }

        }
    }
}
