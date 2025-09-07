#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "collision_and_stream.cu"
#include "initialization.cpp"
#include <iostream> 
#include <cmath>

using namespace std;

#define CUDA_CHECK(call) {                                          \
    do{                                                              \
        cudaError_t e = call;                                        \
        if (e != cudaSuccess) {                                      \
            fprintf(stderr, "Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__, \
                   cudaGetErrorString(e));                           \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0);                                                     \
}

int main() {
    // Initialize simulation parameters on CPU
    setupGridAndCar();

    // Initialize LBM grid on CPU
    float initial_flow_ux = 0.05f; // inlet velocity in x-direction
    float initial_flow_uy = 0.0f;  // inlet velocity in y-direction
    float initial_flow_rho = 1.0f;     // initial density
    initializeLBM(initial_flow_ux, initial_flow_uy, initial_flow_rho);
    cout << "LBM grid initialized on CPU." << endl;

    // Allocate and copy LBM grid to GPU
    LBMCell_GPU* h_lbm_grid = (LBMCell_GPU*)malloc(NX * NY * sizeof(LBMCell_GPU));
    if (!h_lbm_grid) {
        fprintf(stderr, "Failed to allocate host memory for LBM grid.\n");
        exit(EXIT_FAILURE);
    }

    // Copy data from LBMCell to LBMCell_GPU
    for (int i = 0; i < NX * NY; ++i) {
        h_lbm_grid[i].rho = lbm_grid[i].rho;
        h_lbm_grid[i].ux = lbm_grid[i].ux;
        h_lbm_grid[i].uy = lbm_grid[i].uy;
        h_lbm_grid[i].type = lbm_grid[i].type;
        for (int j = 0; j < Q; ++j) {
            h_lbm_grid[i].f[j] = lbm_grid[i].f[j];
            h_lbm_grid[i].f_temp[j] = lbm_grid[i].f_temp[j];
        }
    }

    LBMCell_GPU* d_lbm_grid;
    CUDA_CHECK(cudaMalloc((void**)&d_lbm_grid, NX * NY * sizeof(LBMCell_GPU)));
    CUDA_CHECK(cudaMemcpy(d_lbm_grid, h_lbm_grid, NX * NY * sizeof(LBMCell_GPU), cudaMemcpyHostToDevice));
    cout << "LBM grid copied to GPU." << endl;

    // Set LBM constants in GPU constant memory
    setCUDALBMConstants();
    cout << "LBM constants set in GPU constant memory." << endl;

    // Define CUDA grid and block dimensions
    int threadsPerBlockX = 16;
    int threadsPerBlockY = 16;
    dim3 threadsPerBlock(threadsPerBlockX, threadsPerBlockY);
    dim3 numBlocks((NX + threadsPerBlockX - 1) / threadsPerBlockX,
                   (NY + threadsPerBlockY - 1) / threadsPerBlockY);

    const int NUM_ITERATIONS = 1000; // Number of time steps
    cout << "Starting simulation for " << NUM_ITERATIONS << " iterations..." << endl;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        lbm_kernel<<<numBlocks, threadsPerBlock>>>(d_lbm_grid, NX, NY, initial_flow_ux, initial_flow_uy, initial_flow_rho);
        CUDA_CHECK(cudaGetLastError());

        // Optional: Print progress
        if (iter % 100 == 0) {
            cout << "Completed iteration " << iter << "/" << NUM_ITERATIONS << endl;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cout << "Simulation completed." << endl;

    // Copy results back to CPU
    CUDA_CHECK(cudaMemcpy(h_lbm_grid, d_lbm_grid, NX * NY * sizeof(LBMCell_GPU), cudaMemcpyDeviceToHost));
    cout << "Results copied back to CPU." << endl;

    // Post-process: Copy data back to original LBMCell structure
    cout << "Sample velocities (ux, uy) around the car:\n";
    int car_x_start = NX / 4;
    int car_y_start = NY / 2 - 7;

    for (int y = car_y_start - 5; y <= car_y_start + 5; ++y) {
        for (int x = car_x_start - 5; x <= car_x_start + 45; ++x) {
            int idx = y * NX + x;
            if (h_lbm_grid[idx].type == FLUID) {
                printf("%6.3f ", std::sqrt(h_lbm_grid[idx].ux * h_lbm_grid[idx].ux + h_lbm_grid[idx].uy * h_lbm_grid[idx].uy));
            } else if (h_lbm_grid[idx].type == SOLID) {
                cout << "(X, X) "; // Indicate solid cells
            } else {
                cout << "  ######  "; // Indicate other types (e.g., BOUNDARY)
            }
        }
        cout << endl;
    }

    // Free GPU memory
    free(h_lbm_grid);
    CUDA_CHECK(cudaFree(d_lbm_grid));
    CUDA_CHECK(cudaDeviceReset());
    cout << "GPU memory freed and device reset." << endl;

    return 0;
}