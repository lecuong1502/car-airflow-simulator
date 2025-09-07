#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <bits/stdc++.h>
#include "initialization.cpp"

using namespace std;

// ERROR CHECKING MACRO (Important to debug)
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,        \
               cudaGetErrorString(e));                                \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

struct LBMCell_GPU{
    float f[Q];
    float f_temp[Q];
    float rho;
    float ux, uy;
    int type;
};

// Device constant memory for LBM parameters
__constant__ float d_weights[Q];
__constant__ int d_cx[Q];
__constant__ int d_cy[Q];
__constant__ float d_cs_sq;
__constant__ float d_inv_cs_sq;
__constant__ float d_omega;

// Kernel Collision + Stream
__global__ void lbm_kernel(LBMCell_GPU* d_lbm_grid, int NX, int NY, float inlet_ux, float inlet_uy, float inlet_rho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int current_idx = y * NX + x;
    LBMCell_GPU current_cell = d_lbm_grid[current_idx];

    if (current_cell.type == FLUID) {
        // Compute macroscopic variables
        float current_rho = 0.0f;
        float current_ux = 0.0f;
        float current_uy = 0.0f;
        for (int i = 0; i < Q; ++i) { 
            current_rho += current_cell.f[i];
            current_ux += current_cell.f[i] * d_cx[i];
            current_uy += current_cell.f[i] * d_cy[i];
        }
        current_ux /= current_rho;
        current_uy /= current_rho;

        // Compute equilibrium distribution
        float feq[Q];
        for (int i = 0; i < Q; ++i) {
            float c_dot_u = (float)d_cx[i] * current_ux + (float)d_cy[i] * current_uy;
            float u_sq = current_ux * current_ux + current_uy * current_uy;
            feq[i] = d_weights[i] * current_rho * (1.0f + d_inv_cs_sq * c_dot_u + 0.5f * d_inv_cs_sq * d_inv_cs_sq * c_dot_u * c_dot_u - 0.5f * d_inv_cs_sq * u_sq);
        }

        // Collision step
        for (int i = 0; i < Q; ++i) {
            current_cell.f_temp[i] = current_cell.f[i] + d_omega * (feq[i] - current_cell.f[i]);
        }

        // Streaming step
        for (int i = 0; i < Q; ++i) {
            int prev_x = x - d_cx[i];
            int prev_y = y - d_cy[i];

            // Periodic boundary conditions
            if (prev_x < 0) {  //Left boundary
                current_cell.f[i] = current_cell.f_temp[i];

                float feq_inlet[Q];
                float incoming_rho = inlet_rho;
                float incoming_ux = inlet_ux;
                float incoming_uy = inlet_uy;

                computeEquilibrium(feq_inlet, incoming_rho, incoming_ux, incoming_uy);
                current_cell.f_temp[i] = feq_inlet[i];

            } else if (prev_x >= NX) {  //Right boundary
                current_cell.f[i] = current_cell.f_temp[i];
                float feq_outlet[Q];
                float outgoing_rho = current_rho; // Use local density for outlet
                float outgoing_ux = current_ux;
                float outgoing_uy = current_uy;

                computeEquilibrium(feq_outlet, outgoing_rho, outgoing_ux, outgoing_uy);
                current_cell.f_temp[i] = feq_outlet[i];
            } else if (prev_y < 0 || prev_y >= NY) {  // Top and Bottom boundaries
                current_cell.f[i] = current_cell.f_temp[i];
            } 
            else {  // Internal cells
                int prev_idx = prev_y * NX + prev_x;
                d_lbm_grid[prev_idx].f[i] = current_cell.f_temp[i];
            }
        }
    } else {
        // Bounce-back for solid cells
        for (int i = 0; i < Q; ++i) {
            int opposite_i = (i < 4) ? (i + 4) : (i - 4); // Opposite direction
            current_cell.f[opposite_i] = current_cell.f_temp[i];
        }
    }

    // Update the cell in global memory
    d_lbm_grid[current_idx] = current_cell;
    __syncthreads();
}

// Host function to launch the kernel
void setCUDALBMConstants() {
    cudaCheckError();
    cudaMemcpyToSymbol(d_weights, weights, Q * sizeof(float));
    cudaMemcpyToSymbol(d_cx, cx, Q * sizeof(int));
    cudaMemcpyToSymbol(d_cy, cy, Q * sizeof(int));
    cudaMemcpyToSymbol(d_cs_sq, &cs_sq, sizeof(float));
    cudaMemcpyToSymbol(d_inv_cs_sq, &inv_cs_sq, sizeof(float));
    cudaMemcpyToSymbol(d_omega, &OMEGA, sizeof(float));
    cudaCheckError();

    cout << "LBM constants copied to device.\n" << endl;
}