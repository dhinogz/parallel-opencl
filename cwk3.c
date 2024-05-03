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

void serialCalculateWeights(float *gradients, float *inputs, float *weights, int N, int M) {
	for( int i=0; i<N; i++ )
		for( int j=0; j<M; j++)
			weights[i*M+j] += gradients[i] * inputs[j];
}

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
	
	float
		*serial_gradients = (float*) malloc( N  *sizeof(float) ),
		*serial_inputs    = (float*) malloc(   M*sizeof(float) ),
		*serial_weights   = (float*) malloc( N*M*sizeof(float) );
	initialiseArrays( serial_gradients, serial_inputs, serial_weights, N, M );			// DO NOT REMOVE.

	
	
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

	// create buffer for weights. Nothing would be copied to device since it's the result
	cl_mem device_weights= clCreateBuffer(context, CL_MEM_WRITE_ONLY, N*M*sizeof(float), NULL, &status);

	//
	// Perform calculations on the GPU
	//
	
	// Build kernel code function in cwk3.cl
	cl_kernel kernel = compileKernelFromFile("cwk3.cl", "calculateWeights", context, device);
	
	// Specify args to kernel
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_gradients);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_inputs);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_weights);
	
	size_t globalSize[2] = {N, M};

	// TODO: determine work group max size based on my device
	// size_t workGroupSize[2] = {8,16};

	// Kernel into command queue
	status = clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		2, 
		NULL, 
		globalSize, 
		NULL, // work group size will be calculated automatically
		0, 
		NULL, 
		NULL
	);
	if (status != CL_SUCCESS){
		printf("Failure enqueuing kernel: %d\n", status);
		return EXIT_FAILURE;
	}

	// Get result back from device
	status = clEnqueueReadBuffer(queue, device_weights, CL_TRUE, 0, N*M*sizeof(float), weights, 0, NULL, NULL);
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

	serialCalculateWeights(serial_gradients, serial_inputs, serial_weights, N, M);
	
	displayWeights( serial_weights, N, M) ;								// DO NOT REMOVE.
	// free cl buffers
	free(device_gradients );
	free(device_inputs);
	free(device_weights);

	free( gradients );
	free( inputs    );
	free( weights   );

	// TODO: remove
	free( serial_gradients );
	free( serial_inputs    );
	free( serial_weights   );

	// release kernel
	clReleaseKernel(kernel);

	clReleaseCommandQueue( queue   );
	clReleaseContext     ( context );

	return EXIT_SUCCESS;
}

