//
// Starting point for the GPU coursework. Please read coursework instructions before attempting this.
//


//
// Includes.
//
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include "helper_cwk.h"			// Note this is not the same as the 'helper.h' used for examples.

//
// Main.
//
int main( int argc, char **argv )
{
	//
	// Initialisation.
	//
	
	// Initialise OpenCL. This is the same as the examples in lectures.
	cl_device_id device;
	cl_context context = simpleOpenContext_GPU(&device);

	cl_int status;
	cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );
	
	// Get the parameters (N = no. of nodes/gradients, M = no. of inputs). getCmdLineArgs() is in helper_cwk.h.
	int N, M;
	getCmdLineArgs( argc, argv, &N, &M );

	// Initialise host arrays. initialiseArrays() is defined in helper_cwk.h. DO NOT REMOVE or alter this routine;
	// it will be replaced with a different version as part of the assessment.
	float
		*gradients = (float*) malloc( N  *sizeof(float) ),
		*inputs    = (float*) malloc(   M*sizeof(float) ),
		*weights   = (float*) malloc( N*M*sizeof(float) );
	initialiseArrays( gradients, inputs, weights, N, M );			// DO NOT REMOVE.
	
	//
	// Implement the GPU solution to the problem.
	//
	
	//
	// Alocate memory for the arrays on device
	// 
	
	// create buffer in device for gradients of size N
	cl_mem device_gradients = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(float), gradients, &status);
	
	// create buffer in device for inputs of size M 
	cl_mem device_inputs= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M*sizeof(float), inputs, &status);

	// create buffer for weights. result will be written on this buffer so CL_MEM_READ_WRITE flag is used
	cl_mem device_weights= clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N*M*sizeof(float), weights, &status);

	//
	// Perform calculations on the GPU
	//
	
	// Build kernel code function in cwk3.cl
	cl_kernel kernel = compileKernelFromFile("cwk3.cl", "calculateWeights", context, device);
	
	// Specify args to kernel function
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_gradients);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_inputs);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_weights);
	
	// Global size is mapped to size of matrix NxM
	size_t globalSize[2] = {N, M};

	// size_t maxWorkItems;
	// clGetDeviceInfo(
	// 	device,
	// 	CL_DEVICE_MAX_WORK_GROUP_SIZE,
	// 	sizeof(size_t),
	// 	&maxWorkItems,
	// 	NULL
	// );
	
	// couldn't get the maxWorkItems for ND working
	// size_t maxWorkItemsND[2] = {maxWorkItems/M, maxWorkItems/N};

	// Enqueue command to execute kernel calculateWeights on device
	status = clEnqueueNDRangeKernel(
		queue,
		kernel, 
		2, // work is two-dimensional
		NULL,
		globalSize,
		// maxWorkItemsND,
		NULL,
		0,
		NULL,
		NULL
	);
	if (status != CL_SUCCESS){
		printf("Failure enqueuing kernel: %d\n", status);
		return EXIT_FAILURE;
	}

	// Get result back from device by reading device_weights buffer
	status = clEnqueueReadBuffer(
		queue,
		device_weights, // buffer to be read
		CL_TRUE,
		0,
		N*M*sizeof(float),
		weights, // buffer to write to
		0,
		NULL,
		NULL
	);
	if (status != CL_SUCCESS) {
		printf("Could not copy data to host: %d\n", status);
		return EXIT_FAILURE;
	}

	//
	// Output the result and clear up.
	//
	
	// Output result to screen. DO NOT REMOVE THIS LINE (or alter displayWeights() in helper_cwk.h); this will be replaced
	// with a different displayWeights() for the the assessment, so any changes you might make will be lost.
	displayWeights( weights, N, M) ;								// DO NOT REMOVE.
	
	// free device buffers
	free(device_gradients);
	free(device_inputs);
	free(device_weights);

	free( gradients );
	free( inputs    );
	free( weights   );

	// release kernel
	clReleaseKernel(kernel);
	clReleaseCommandQueue( queue   );
	clReleaseContext     ( context );

	return EXIT_SUCCESS;
}

