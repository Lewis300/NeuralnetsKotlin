__global__ void matmul(float *a, float *b, float *c, int n) {
    // compute each thread's row
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // compute each thread's column
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    if((row < n) && (col < n)) {

        // Iterate of row, and down column
        for(int k = 0; k < n; k++) {
            // Accumulate result for a single element
            temp_sum += a[row*n + k] * b[k*n + col];
        }
        // Assign result
        c[row*n + col] = temp_sum;
    }
}