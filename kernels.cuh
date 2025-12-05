__global__ void initRowPointers(vector3** accels, vector3* values, int N);
__global__
void pairwise_comp(vector3** accels, int N, vector3* dPos, double* dMass);
__global__
void row_sum(vector3** accels, int N, vector3* dPos, vector3* dVel);
