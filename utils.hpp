#pragma once
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

// const int Q = 9;
// const float weights[Q] = { 4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f };
// const int cx[Q] = {0,1,0,-1,0,1,-1,-1,1};
// const int cy[Q] = {0,0,1,0,-1,1,1,-1,-1};

extern const int Q;
extern const float weights[];
extern const int cx[];
extern const int cy[];

void computeEquilibrium2(float *feq, float rho, float ux, float uy) {
    const float cs_sq = 1.0f/3.0f;
    const float inv_cs_sq = 1.0f/cs_sq;

    for (int i = 0; i < Q; ++i) {
        float c_dot_u = (float)cx[i] * ux + (float)cy[i] * uy;
        float u_sq = ux*ux + uy*uy;
        feq[i] = weights[i] * rho *
            (1.0f + inv_cs_sq * c_dot_u
                   + 0.5f * inv_cs_sq * inv_cs_sq * c_dot_u * c_dot_u
                   - 0.5f * inv_cs_sq * u_sq);
    }
}