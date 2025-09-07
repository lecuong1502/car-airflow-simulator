#include <vector>
#include <iostream>
#include <bits/stdc++.h>

using namespace std;

// Size of grid
const int NX = 200;     // X-dimension: amount of points
const int NY = 100;     // Y-dimension: amount of points

// States of a grid cell
enum CellType { FLUID, SOLID };

// CellType grid to display the space and solid
vector<CellType> grid(NX * NY);

int car_x_start;
int car_y_start;
int car_width;
int car_height;

// Set CellType in (x, y) 
CellType getCellType(int x, int y) {
    if (x < 0 || x >= NX || y < 0 || y > NY) {
        return SOLID;           // Solid is out of the edge
    }
    
    return grid[y * NX + x];
}

void setCellType(int x, int y, CellType type) {
    if (x >= 0 && x < NX && y >= 0 && y < NY) {
        grid[y * NX + x] = type;
    }
}

// Initialize the grid and set the car (rectangle block)
void setupGridAndCar() {
    for (int i = 0; i < NX * NY; ++i) {
        grid[i] =  FLUID;
    }

    car_width = 40;
    car_height = 15;
    car_x_start = NX / 4;
    car_y_start = NY / 2 - car_height / 2;

    for (int y = car_y_start; y < car_y_start + car_height; ++y) {
        for (int x = car_x_start; x < car_x_start + car_width; ++x) {
            setCellType(x, y, SOLID);
        }
    }
    
    cout << "Grid and car initialized.\n";
}

int main_grid_setup() {
    setupGridAndCar();

    for (int y = NY / 2 - 10; y < NY / 2 + 10; ++y) {
        for (int x = car_x_start - 10; x < car_x_start + car_width + 10; ++x) {
            std::cout << (getCellType(x, y) == FLUID ? '.' : '#') << " ";
        }
        std::cout << "\n";
    }

    return 0;
}