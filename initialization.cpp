// Quantities to be initialized
// Paricle distribution function (PDFs) f[9]: these values will change over time
// rho (local density): Sum of 9 PDFs
// ux, uy (local velocities): from PDFs
// weights[9] (weight constant): Fixed for D2Q9 model
// cx[9], cy[9] (velocity vectors): 9 fixed directions

#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "2Dgrid.cpp"
#include "utils.hpp"

using namespace std;

const int Q = 9;
const float W0 = 4.0f / 9.0f;
const float W1 = 1.0f / 9.0f;
const float W2 = 1.0f / 36.0f;

const float weights[Q] = {
    W0, W1, W1, W1, W1, W2, W2, W2, W2
};

const int cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

// Speed ​​of sound squared (sq(cs))
const float cs_sq = 1.0f / 3.0f;
const float inv_cs_sq = 1.0f / cs_sq;

// Relaxation time constants (Between 0.5 and 2.0)
const float OMEGA = 1.0f;           // tau = 1/omega

// Data for each grid cell
struct LBMCell {
    float f[Q];       // Distribution functions
    float f_temp[Q];  // Temporary distributions for streaming
    float rho;        // Local density
    float ux, uy;     // Local velocity
    CellType type;    // FLUID or SOLID
};

// LBMCell grid
vector<LBMCell> lbm_grid(NX * NY);

// // Equilibrium distribution function
// void computeEquilibrium(float *feq, float rho, float ux, float uy) {
//     for (int i = 0; i < Q; ++i) {
//         float c_dot_u = (float)cx[i] * ux + (float)cy[i] * uy;
//         float u_sq = ux * ux + uy * uy;
//         feq[i] = weights[i] * rho * (1.0f + inv_cs_sq * c_dot_u + 0.5f * inv_cs_sq * inv_cs_sq * c_dot_u * c_dot_u - 0.5f * inv_cs_sq * u_sq);
//     }   
// }

// Initialize all the system
void initializeLBM(float initial_ux, float initial_uy, float initial_rho) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            lbm_grid[idx].type = getCellType(x, y);
            // Take the grid type

            if (lbm_grid[idx].type == FLUID){
                lbm_grid[idx].rho = initial_rho;
                lbm_grid[idx].ux = initial_ux;
                lbm_grid[idx].uy = initial_uy;
                computeEquilibrium(lbm_grid[idx].f, initial_rho, initial_ux, initial_uy);

                // Copy to first f_temp
                for (int i = 0; i < Q; ++i) {
                    lbm_grid[idx].f_temp[i] = lbm_grid[idx].f[i];
                }
            } else {         // SOLID cells
                lbm_grid[idx].rho = initial_rho;
                lbm_grid[idx].ux = 0.0f;
                lbm_grid[idx].uy = 0.0f;

                for (int i = 0; i < Q; ++i) {
                    lbm_grid[idx].f[i] = 0.0f;
                    lbm_grid[idx].f_temp[i] = 0.0f;
                }
            }
        }
    }

    cout << "LBM initialized with initial flow.\n";
}

int main_lbm_init() {
    main_grid_setup();
    initializeLBM(0.05f, 0.0f, 1.0f); // Flow runs from left to right with the velocity = 0.05
    return 0;
}