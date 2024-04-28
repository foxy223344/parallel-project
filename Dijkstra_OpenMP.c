
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define INFINITY 1000000

int Find_min_dist(int loc_dist[], int loc_known[], int loc_n) {
    int loc_u = -1, loc_v;
    int shortest_dist = INFINITY;

    #pragma omp parallel for shared(loc_known, loc_dist, shortest_dist) private(loc_v)
    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        if (!loc_known[loc_v]) {
            #pragma omp critical
            {
                if (loc_dist[loc_v] < shortest_dist) {
                    shortest_dist = loc_dist[loc_v];
                    loc_u = loc_v;
                }
            }
        }
    }
    return loc_u;
}

void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n) {
    int loc_u, loc_v, new_dist;
    int *loc_known;

    loc_known = (int*)malloc(loc_n * sizeof(int));

    #pragma omp parallel for shared(loc_known) private(loc_v)
    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        loc_known[loc_v] = 0;
    }

    #pragma omp parallel for shared(loc_dist, loc_pred) private(loc_v)
    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        loc_dist[loc_v] = loc_mat[0 * loc_n + loc_v];
        loc_pred[loc_v] = 0;
    }

    for (int i = 0; i < n - 1; i++) {
        loc_u = Find_min_dist(loc_dist, loc_known, loc_n);

        if (loc_u != -1) {
            #pragma omp parallel for shared(loc_known, loc_dist, loc_pred, loc_u) private(loc_v, new_dist)
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

        if (loc_u != -1) {
            loc_known[loc_u] = 1;
        }
    }

    free(loc_known);
}

int main(int argc, char **argv) {
    int *loc_mat, *loc_dist, *loc_pred;
    int p, loc_n, n;

    p = omp_get_max_threads();
    n = Read_n();

    loc_n = n / p;
    loc_mat = (int*)malloc(n * loc_n * sizeof(int));
    loc_dist = (int*)malloc(loc_n * sizeof(int));
    loc_pred = (int*)malloc(loc_n * sizeof(int));

    if (my_rank == 0) {
        Read_matrix(loc_mat, n, loc_n);
    }

    Dijkstra(loc_mat, loc_dist, loc_pred, loc_n, n);

    free(loc_mat);
    free(loc_dist);
    free(loc_pred);
    return 0;
}