/* CC: nvcc -g */
/* lib_list: -lmpi */
/* run: mpirun -l -n 2 */

#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdlib.h>


#define m_cudart_check_errors(res)                                                              \
    do {                                                                                        \
        cudaError_t __err = res;                                                                \
        if (__err != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA RT error %s at %s:%d\n", cudaGetErrorString(__err), __FILE__, \
                    __LINE__);                                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int mpi_errno;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Process %d / %d\n", rank, size);


    cudaStream_t stream;
    cudaStreamCreate(&stream);

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "type", "cudaStream_t");
    //MPI_Info_set(info, "value", &stream);
    //MPI_Info_set(info, "id", str_stream);
    MPIX_Info_set_hex(info, "value", &stream, sizeof(cudaStream_t));

    MPIX_Stream mpi_stream;
    MPIX_Stream_create(info, &mpi_stream);

    MPI_Comm stream_comm;
    MPIX_Stream_comm_create(MPI_COMM_WORLD, mpi_stream, &stream_comm);

    MPI_Request send_req;

    printf("Init done, stream_comm created\n");

    for (int i = 0; i <= 26; i++) {
        long int N = 1 << i;

        double *x, *y, *d_x, *d_y;
        x = (double*)malloc(N*sizeof(double));
        //y = (double*)malloc(N*sizeof(double));
    
        for (int j = 0; j < N; j++) {
            x[j] = 3.0;
            //y[j] = 4.0;
        }
        m_cudart_check_errors(cudaMalloc(&d_x, N*sizeof(double)));
        m_cudart_check_errors(cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice));
        double tim_x = MPI_Wtime();
        //m_cudart_check_errors(cudaMalloc(&d_y, N*sizeof(double)));
        const int loop_count = 25;

        double start_time, stop_time, elapsed_time = 0.0;

        for (int i = -5; i < loop_count; i++) {
            MPI_Request recv_req;
            if (rank == 0) {  
                cudaMemcpyAsync(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice, stream);
                mpi_errno = MPIX_Irecv_enqueue(d_x, N, MPI_DOUBLE, 1, 1, stream_comm, &recv_req);
                assert(mpi_errno == MPI_SUCCESS);
                MPI_Barrier(MPI_COMM_WORLD);  // barrier is on world comm
                start_time = MPI_Wtime();
                mpi_errno = MPIX_Send_enqueue(d_x, N, MPI_DOUBLE, 1, 0, stream_comm);
                assert(mpi_errno == MPI_SUCCESS);     
                MPIX_Wait_enqueue(&recv_req, MPI_STATUS_IGNORE);
                stop_time = MPI_Wtime();
                // only count the non-warm-up loops
                if (i >= 0){
                    elapsed_time += stop_time - start_time;
                }
            } else if (rank == 1) {
                cudaMemcpyAsync(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice, stream);
                mpi_errno = MPIX_Irecv_enqueue(d_x, N, MPI_DOUBLE, 0, 0, stream_comm, &recv_req);
                assert(mpi_errno == MPI_SUCCESS);
                MPI_Barrier(MPI_COMM_WORLD);  // barrier is on world comm
                MPIX_Wait_enqueue(&recv_req, MPI_STATUS_IGNORE);
                mpi_errno = MPIX_Send_enqueue(d_x, N, MPI_DOUBLE, 0, 1, stream_comm);
                assert(mpi_errno == MPI_SUCCESS);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        //printf("Time for one loop of 50 send/recv: %f seconds\n", MPI_Wtime() - tim_x);
        //m_cudart_check_errors(cudaStreamSynchronize(stream));
        m_cudart_check_errors(cudaFree(d_x));
        m_cudart_check_errors(cudaFree(d_y));
        free(x);
        free(y);
        if (0 == rank) {
            long int num_B                 = N * sizeof(double);
            long int B_in_GB               = 1 << 30;
            double   num_GB                = (double)num_B / (double)B_in_GB;
            double   avg_time_per_transfer = elapsed_time / (2.0 * (double)loop_count);

            printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB / avg_time_per_transfer);
        }
    }
    //cudaFree(d_x);
    //cudaFree(d_y);
    //free(x);
    //free(y);

    cudaStreamDestroy(stream);
    MPI_Finalize();
}
