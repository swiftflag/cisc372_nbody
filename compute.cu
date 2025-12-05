#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(vector3* dPos, vector3* dVel, double* dMass){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	vector3* values;
	vector3** accels;

	cudaMalloc(&values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMalloc(&accels, sizeof(vector3*)*NUMENTITIES);

	int blockSize = 256;
	int numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

	initRowPointers<<<numBlocks, blockSize>>>(accels, values, NUMENTITIES);
	//first compute the pairwise accelerations.  Effect is on the first argument.
	dim3 block(16, 16);   // 16 threads in x, 16 in y
	dim3 grid((NUMENTITIES + 15) / 16, (NUMENTITIES + 15) / 16);
	
	pairwise_comp<<<grid, block>>>(accels, NUMENTITIES, dPos, dMass);       
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
        row_sum<<<numBlocks, blockSize>>>(accels, NUMENTITIES, dPos, dVel);	
	cudaDeviceSynchronize();
	cudaFree(accels);
	cudaFree(values);
}
