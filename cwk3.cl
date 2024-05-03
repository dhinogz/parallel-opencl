// Implement the kernel (or kernels) for coursework 3 in this file.
__kernel
void calculateWeights(__global float *gradients, __global float *inputs, __global float *weights) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int m = get_global_size(1);

    printf("index: %.6f\ni: %d gradient: %.6f\nj: %d input: %.6f\n", i*m+j, i, gradients[i], j, inputs[j]);

    weights[i*m+j] += gradients[i] * inputs[j];
}
