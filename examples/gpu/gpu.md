
In this context, CPU is the host and the GPU is the device

Host -1-> 2 Device -3-> Host
1. Send problem data from host to device
2. Perform calculation on GPU
3. Return results

To use OpenCL, we need to:
- identify the platform and a suitable device
- for each device initialise
    - context
    - command queue
- We can use the helper function `simpleOpenContext_GPU()`

```c
cl_device_id device;
cl_context context = simpleOpenContext_GPU(&device);

cl_int status;
cl_command_queue queue = clCreateCommandQueue(context, device, 0, %status);

// Use GPU here

clReleaseCommandQueue(queue);
clReleaseContext(context);
```

To allocate device memory
```c
// initialise array on host
float *host_a = (float*) malloc(N*sizeof(float));

// allocate memory to device
cl_mem device_a = clCreateBuffer(
    // Created using simpleOpenContext_GPU function
    context, 

    // flags, how the device access memory
    // read only optimises runtime execution
    // memory copy host pointer makes sure to copy from 4th argument
    // if flag dne, 4th arg is NULL
    CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,     

    // size in bytes
    N*sizeof(float), 

    // Copy into buffer in device
    host_a, 

    // Error status
    &status 
);
```

GPU kernel contains functions that execute on device
Each thread within every SIMD core execute the kernel

- kernel functions:
    - procede with kernel
    - must return void
    - __global refers to device memory we have allocated
    - get_global_id() return global index for the 'this' thread
        - parallelised solution, every index in vector is it's own thread
```c
__kernel
void vectorAdd(__global float a, __global float *b, __global float *c) {
    // gets currrent thread's ID 
    int gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}
```

Building the kernel functions 
- complicated steps
- simplified by helper function `compileKernelFromFile()`

```c
cl_kernel kernel = compileKernelFromFile(
    "file_with_kernel_code.cl",
    "function_name",
    context,
    device
);

// Use kernel function

clReleaseKernel(kernel);
```
