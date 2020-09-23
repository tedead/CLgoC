#pragma comment(lib, "OpenCL.lib")
#define CL_TARGET_OPENCL_VERSION 220
#define _CRT_SECURE_NO_DEPRECATE
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
//#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <fstream>
#include <CL/cl.h>
//#include "opencl.hpp"
#define MAX_SOURCE_SIZE (0x100000)
FILE *stream, *stream2;

int main(void) {

	errno_t err;

	err = fopen_s(&stream2, "results.txt", "w+");

	if (err != 0) {
		fprintf(stderr, "Failed to load file.\n");
		exit(1);
	}

	fprintf(stream2, "***** Detected Device Information *****\n");

	//DETECT & RECORD DEVICES
	char* value_i;
	size_t valueSize_i;
	cl_uint platformCount_i;
	cl_platform_id* platforms_i;
	cl_uint deviceCount_i;
	cl_device_id* devices_i;
	cl_uint maxComputeUnits_i;

	// get all platforms
	clGetPlatformIDs(0, NULL, &platformCount_i);
	platforms_i = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount_i);
	clGetPlatformIDs(platformCount_i, platforms_i, NULL);

	for (unsigned int i = 0; i < platformCount_i; i++) {

		// get all devices
		clGetDeviceIDs(platforms_i[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount_i);
		devices_i = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount_i);
		clGetDeviceIDs(platforms_i[i], CL_DEVICE_TYPE_ALL, deviceCount_i, devices_i, NULL);

		// for each device print critical attributes
		for (unsigned int j = 0; j < deviceCount_i; j++) {

			// print device name
			clGetDeviceInfo(devices_i[j], CL_DEVICE_NAME, 0, NULL, &valueSize_i);
			value_i = (char*)malloc(valueSize_i);
			clGetDeviceInfo(devices_i[j], CL_DEVICE_NAME, valueSize_i, value_i, NULL);
			printf("%d. Device: %s\n", j + 1, value_i);
			fprintf(stream2, "%d. Device: %s\n", j + 1, value_i);
			free(value_i);

			clGetDeviceInfo(devices_i[j], CL_DEVICE_BUILT_IN_KERNELS, 0, NULL, &valueSize_i);
			value_i = (char*)malloc(valueSize_i);
			clGetDeviceInfo(devices_i[j], CL_DEVICE_BUILT_IN_KERNELS, valueSize_i, value_i, NULL);
			printf("%d. Device Kernels %s\n", j + 1, value_i);
			fprintf(stream2, "%d. Device Kernels: %s\n", j + 1, value_i);
			free(value_i);

			clGetDeviceInfo(devices_i[j], CL_DEVICE_VENDOR_ID, 0, NULL, &valueSize_i);
			value_i = (char*)malloc(valueSize_i);
			clGetDeviceInfo(devices_i[j], CL_DEVICE_VENDOR_ID, valueSize_i, value_i, NULL);
			printf("%d. Device Vendor: %s\n", j + 1, value_i);
			fprintf(stream2, "%d. Device Vendor: %s\n", j + 1, value_i);
			free(value_i);

			// print hardware device version
			clGetDeviceInfo(devices_i[j], CL_DEVICE_VERSION, 0, NULL, &valueSize_i);
			value_i = (char*)malloc(valueSize_i);
			clGetDeviceInfo(devices_i[j], CL_DEVICE_VERSION, valueSize_i, value_i, NULL);
			printf("%d.%d Hardware version: %s\n", j + 1, 1, value_i);
			fprintf(stream2, "\t%d.%d Hardware version: %s\n", j + 1, 1, value_i);
			free(value_i);


			clGetDeviceInfo(devices_i[j], CL_DEVICE_COMPILER_AVAILABLE, 0, NULL, &valueSize_i);
			value_i = (char*)malloc(valueSize_i);
			clGetDeviceInfo(devices_i[j], CL_DEVICE_COMPILER_AVAILABLE, valueSize_i, value_i, NULL);
			printf("%d.%d Compiler: %s\n", j + 1, 1, value_i);
			fprintf(stream2, "\t%d.%d Compiler: %s\n", j + 1, 1, value_i);
			free(value_i);


			// print software driver version
			clGetDeviceInfo(devices_i[j], CL_DRIVER_VERSION, 0, NULL, &valueSize_i);
			value_i = (char*)malloc(valueSize_i);
			clGetDeviceInfo(devices_i[j], CL_DRIVER_VERSION, valueSize_i, value_i, NULL);
			printf(" %d.%d Software version: %s\n", j + 1, 2, value_i);
			fprintf(stream2, "%d.%d Software version: %s\n", j + 1, 2, value_i);
			free(value_i);

			// print c version supported by compiler for device
			clGetDeviceInfo(devices_i[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize_i);
			value_i = (char*)malloc(valueSize_i);
			clGetDeviceInfo(devices_i[j], CL_DEVICE_OPENCL_C_VERSION, valueSize_i, value_i, NULL);
			printf(" %d.%d OpenCL C version: %s\n", j + 1, 3, value_i);
			fprintf(stream2, "%d.%d OpenCL C version: %s\n", j + 1, 3, value_i);
			free(value_i);

			// print parallel compute units
			clGetDeviceInfo(devices_i[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits_i), &maxComputeUnits_i, NULL);
			printf(" %d.%d Parallel compute units: %d\n", j + 1, 4, maxComputeUnits_i);
			fprintf(stream2, "%d.%d Parallel compute units: %d\n", j + 1, 4, maxComputeUnits_i);

		}

		free(devices_i);

	}

	free(platforms_i);
	//return 0;

	printf("started running\n");

	// Create the two input vectors
	int i;
	const int LIST_SIZE = 16; //1024;
	int* A = (int*)malloc(sizeof(int) * LIST_SIZE);
	int* B = (int*)malloc(sizeof(int) * LIST_SIZE);

	for (i = 0; i < LIST_SIZE; i++) {
		A[i] = i;
		B[i] = LIST_SIZE - i;
	}

	char* source_str;
	size_t source_size;

	err = fopen_s(&stream, "kernel.cl", "r");

	if (err != 0) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}

	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, stream);

	//fclose(fp);
	if (stream) {
		err = fclose(stream);
	}

	printf("kernel loading done\n");
	fprintf(stream2, "kernel loading complete\n");
	// Get platform and device information
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;


	cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);

	printf("clGetPlatformIDs at code line %d is %d\n", __LINE__, ret_num_platforms);

	cl_platform_id* platforms = NULL;
	platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

	ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
	printf("ret at %d is %d\n", __LINE__, ret);

	for (unsigned int x = 0; x < ret_num_platforms; x++) {
		ret = clGetDeviceIDs(platforms[x], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
		printf("ret at %d is %d\n", __LINE__, ret);
		printf("ret at %d is %d\n", __LINE__, platforms[x]);
		printf("device_id at %d is %d\n", __LINE__, device_id);
	}


	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	printf("ret at %d is %d\n", __LINE__, ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	printf("ret at %d is %d\n", __LINE__, ret);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		LIST_SIZE * sizeof(int), NULL, &ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
		LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
	printf("ret at %d is %d\n", __LINE__, ret);

	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
		LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
	printf("ret at %d is %d\n", __LINE__, ret);

	printf("before building\n");
	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
		(const char**)&source_str, (const size_t*)&source_size, &ret);
	printf("ret at %d is %d\n", __LINE__, ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	printf("ret at %d is %d\n", __LINE__, ret);

	printf("after building\n");
	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
	printf("ret at %d is %d\n", __LINE__, ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
	printf("ret at %d is %d\n", __LINE__, ret);

	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
	printf("ret at %d is %d\n", __LINE__, ret);

	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);
	printf("ret at %d is %d\n", __LINE__, ret);

	//added this to fix garbage output problem
	//ret = clSetKernelArg(kernel, 3, sizeof(int), &LIST_SIZE);

	printf("before execution\n");
	// Execute the OpenCL kernel on the list
	size_t global_item_size = LIST_SIZE; // Process the entire lists
	size_t local_item_size = 64; // Divide work items into groups of 64

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
	printf("after execution\n");

	// Read the memory buffer C on the device to the local variable C
	int* C = (int*)malloc(sizeof(int) * LIST_SIZE);
	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
	printf("after copying\n");

	// Display the result to the screen
	for (i = 0; i < LIST_SIZE; i++) {
		printf("%d + %d = %d\n", A[i], B[i], C[i]);
	}

	








	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	return 0;


	if (stream2) {
		err = fclose(stream2);
	}

}

















// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
