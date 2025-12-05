#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "vector.h"
__global__ void initRowPointers(vector3** accels, vector3* values, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        accels[i] = values + i * N;
    }
}
__global__
void pairwise_comp(vector3** accels, int N, vector3* dPos, double* dMass){
	int j = blockIdx.x * blockDim.x + threadIdx.x; // column
   	int i = blockIdx.y * blockDim.y + threadIdx.y; // row

    	if (i >= N || j >= N) return;
	if (i==j) {
		FILL_VECTOR(accels[i][j],0,0,0);
        }
        else{
		vector3 distance;
                for (int k=0;k<3;k++) distance[k]=dPos[i][k]-dPos[j][k];
                double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
                double magnitude=sqrt(magnitude_sq);
                double accelmag=-1*GRAV_CONSTANT*dMass[j]/magnitude_sq;
  		FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
 }
}

__global__
void row_sum(vector3** accels, int N, vector3* dPos, vector3* dVel){
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
         if (i >= N) return;
	
	vector3 accel_sum={0,0,0};
        for (int j=0;j<N;j++){
		for (int k=0;k<3;k++)
                	accel_sum[k]+=accels[i][j][k];
                }
                //compute the new velocity based on the acceleration and time interval
                //compute the new position based on the velocity and time interval
        for (int k=0;k<3;k++){
                dVel[i][k]+=accel_sum[k]*INTERVAL;
                dPos[i][k]+=dVel[i][k]*INTERVAL;
        }
}

