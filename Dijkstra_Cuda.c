
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define INFINITY 1000000

__device__ int Find_min_dist(int loc_dist[], int loc_known[], int loc_n) {
    int loc_u = -1, loc_v;
    int shortest_dist = INFINITY;

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        if (!loc_known[loc_v]) {
            if (loc_dist[loc_v] < shortest_dist) {
                shortest_dist = loc_dist[loc_v];
                loc_u = loc_v;
            }
        }
    }
    return loc_u;
}

__global__ void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n) {
    int loc_u, loc_v, new_dist;
    int *loc_known;

    loc_known = (int*)malloc(loc_n * sizeof(int));

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            loc_known[loc_v] = 1;
        else
            loc_known[loc_v] = 0;
    }

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        loc_dist[loc_v] = loc_mat[0 * loc_n + loc_v];
        loc_pred[loc_v] = 0;
    }

    for (int i = 0; i < n - 1; i++) {
        loc_u = Find_min_dist(loc_dist, loc_known, loc_n);

        if (loc_u != -1) {
            for (loc_v = 0; loc_v < loc_n; loc_v++) {
                if (!loc_known[loc_v]) {
                    new_dist = loc_dist[loc_u] + loc_mat[loc_u * loc_n + loc_v];
                    if (new_dist < loc_dist[loc_v]) {
                        loc_dist[loc_v] = new_dist;
                        loc_pred[loc_v] = loc_u;
                    }
                }
            }
        }

        __syncthreads();

        if (blockIdx.x == 0) {
            loc_u = Find_min_dist(loc_dist, loc_known, loc_n);
            if (loc_u != -1) {
                loc_known[loc_u] = 1;
            }
        }

        __syncthreads();
    }
}

int main(int argc, char **argv) {
    int *loc_mat, *loc_dist, *loc_pred;
    int p, loc_n, n;
    MPI_Comm comm;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    n = Read_n(comm);
    loc_n = n / p;
    loc_mat = (int*)malloc(n * loc_n * sizeof(int));
    loc_dist = (int*)malloc(loc_n * sizeof(int));
    loc_pred = (int*)malloc(loc_n * sizeof(int));

    if (my_rank == 0) {
        Read_matrix(loc_mat, n, loc_n, comm);
    }

    int *d_loc_mat, *d_loc_dist, *d_loc_pred;
    cudaMalloc((void**)&d_loc_mat, n * loc_n * sizeof(int));
    cudaMalloc((void**)&d_loc_dist, loc_n * sizeof(int));
    cudaMalloc((void**)&d_loc_pred, loc_n * sizeof(int));
    cudaMemcpy(d_loc_mat, loc_mat, n * loc_n * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = p;
    int threads_per_block = loc_n;
    Dijkstra<<<blocks, threads_per_block>>>(d_loc_mat, d_loc_dist, d_loc_pred, loc_n, n);

    cudaMemcpy(loc_dist, d_loc_dist, loc_n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(loc_pred, d_loc_pred, loc_n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_loc_mat);
    cudaFree(d_loc_dist);
    cudaFree(d_loc_pred);

    free(loc_mat);
    free(loc_dist);
    free(loc_pred);
    MPI_Finalize();
    return 0;
}
