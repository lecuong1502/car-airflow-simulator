# car-airflow-simulator

- I use a simple method, it is Lattice Boltzman Method (LBM), because LBM easily parallelizes on GPU and good enough to simulate basic airflow.
- LBM is a computional fluid dynamics (CFD) technique that simulates fluid flow by modeling discrete particles on a lattice, rather than solving macroscopic Navier-Stokes equations directly. 

- How it works:
1. Discrete Lattice and Particles:
    + The fluid is represented by a lattice of discrete points (nodes). At each node, a set of distribution functions (denoted as f) represents the density of particles moving with specific discrete velocities.
2. Streaming (Advection):
    + In each time step, the particle distributions move from their node to a neighboring node according to their discrete velocities.
3. Collision:
    + After streaming, particles at the same node interact and collide, changing their velohicles and redistributing according to a collision model, such as the Bhatnagar-Gross-Krook (BGK) model. This collision step brings the system towards local equilibrium.
4. Macroscopic Quantities:
    + By averaging these distribution functions over the velocity space at each node, macroscopic fluid properties such as density and momentum (which can be related to velocity) are obtained.

- Advantages of LBM:
1. Complex Geometries:
    + LBM inherently handles complex boundaries and geometric features without the need for complex meshing, which is a significant advantage over traditional CFD methods.
2. Multiphysics:
    + The method can be extended to incorporate additional physical phenomena, such as heat transfer, turbulence and multiphase flows, with relaative ease.
3. Parallel Computing:
    + Its design is well-suited for parallel processing and high-performance computing, allowing for efficient simulations on modern hardware like GPUs.
4. Mesoscopic Scale:
    + By briding microscopic particle behavior with macroscopic fluid properties, LBM offers a unique approach to understanding fluid dynamics. 

- Architecture:
    + 3D model and Mesh: perform the car and surrounding space.
    + Initialization: Init the first conditions for flows
    + CUDA kernel for LBM (Collision & Stream): 2 main steps of LBM.
    + Boundary Conditions: process the interaction to the surface of car and the edges of computing domains.
    + Time Integration: loops of simulating steps.
    +  Post-processing & Visualization: perform the result.

1. 3D model and Mesh: (2Dgrid.cpp)
    - I simplify the car to a simple geometric block in a 2D grid. Each point has relative attributions to flows. The car is displayed by a set of grid points which is marked "solid". 
    - Code in: C++ (CPU)

2. Initialization for LBM: (initialization.cpp)
    - I follow the particle distribution functions (PDFs) and move according to the discrete direction. In D2Q9 model (2D with 9 directions), each grid point has 9 PDFs.
    - Code in: C++ (CPU)

3. CUDA kernel for LBM (Collision & Stream):
    - Collision: At each FLUID grid point, the current PDFs f are adjusted to move closer to the equilibrium distribution feq.
    - Stream: PDFs, which are adjusted, are moved to neighbor grid points in their direction.
    - I created 2 kernel: Collision and Stream in CUDA.

4. Boundary Conditions: (code in collision_andstreaming.cu)
    - Inlet (Left side): Where the gas enters. Usually a constant velocity flow is applied (e.g. initial_ux).
    - Outlet (Right side): Where the gas escapes. Often uses conditions such as 0th order Extrapolation or Neumann to allow for a smooth escape flow.
    - Top/Bottom side: Usually "slip" or "bounce-back" for limited wind tunnel simulations.
    - Solid (Car): Particles colliding with an obstacle surface will "bounce-back". This is the simplest way to model a solid surface in LBM.

5. Time Integration: (time.cu)
    - This is the main loop of simulation, it calls CUDA kernel again.

6. Post-processing & Visualization:
    - Save the results: After the simulation is complete, I save the rho, ux, uy fields to a file (e.g. .csv, .dat, or a specialized format like .vtu for VTK).
    - Using external software: Software like ParaView or VisIt are powerful for visualizing CFD data. They can generate color maps, velocity vectors, streamlines to understand the airflow.
    - Simple visualization on the console: As in the main above, I print out a small portion of the data for a quick check.
    - Graphics libraries: With C++, I integrate OpenGL or SFML to draw vectors or color maps directly. However, this is more complicated and is usually beyond the scope of a basic LBM example.



- Run the program:
    + Compiling command Terminal: nvcc time.cu -o lbm_car
    + Running command Terminal: ./lbm_car
    + OS of my computer: Ubuntu 24.04
    + GPU: NVIDIA GeForce RTX 4050
    + CPU: 13th Gen Intel(R) Core(TM) i7-13650HX
    + CUDA Toolkit 12.9.