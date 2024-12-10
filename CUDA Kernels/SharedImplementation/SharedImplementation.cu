
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <ctime>

#include "HelperFunctions.h"
#include "ErrorChecker.cuh"

#define NUMBER_OF_CYCLES 1000
#define TILE_WIDTH 16

const unsigned int SEED_VALUE = 2024;
const bool DRY_RUN = false;

cudaError_t nbodyHelperFunction(MassObject** allArrs, int* remainingObjs, int px, int py, int stepsize);

__global__ void calculateSharedAcc(float3* pos) {
    //x and y are the 2D positions and z is the weight of the particle  
    extern __shared__ float3 shPosition[];
    float3* globalPos = (float3*)pos;
    float2* globalAcc = (float2*)pos;

    int i, j, tile;
    float2 acc = { 0.0f, 0.0f };
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //operates tile by tile for optimization (may change logic this later on)
    for (i = 0, tile = 0; i < N; i += TILE_WIDTH, tile++)
    {
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalPos[idx];
        __syncthreads();

        //acceleration calculation
        for (j = 0; j < blockDim.x; j++) {
            float2 vec;

            //vector from current particle to its computational partner particle
            vec.x = pos[gtid].x - pos[j].x;
            vec.y = pos[gtid].y - pos[j].y;
            //distance squared calculation
            float sqrddist = vec.x * vec.x + vec.y * vec.y;

            if (sqrddist > 0) {
                //net_acc  from this object
                float net_acc = pos[j].z / sqrddist;

                //increment acceleration
                acc.x += cosf(atan2f(vec.y, vec.x)) * net_acc;
                acc.y += sinf(atan2f(vec.y, vec.x)) * net_acc;
            }
        }
    }
    globalAcc[gtid] = acc;
}

//semi-randomly initialize the MassObjects given the field size and the number of objects
//all objects are randomly initialized with a mass between 10^22 kg to 10^24 kg
//40% of the objects will be initialized in a central 2.5*10^10 by 2.5*10^10 field
//The remaining 60% can spawn anywhere in the frame's field
void init(int px, int pz, int numberOfObjects, MassObject* arr) {
    int benchmark1 = numberOfObjects * 4 / 10;
    for (int i = 0; i < benchmark1; i++) {
        float x = (0.5 + randfloat(0, 2.5)) * (float)pow(10, 10);
        float y = (0.5 + randfloat(0, 2.5)) * (float)pow(10, 10);
        float vx = rand() % (500) - 250.;
        vx *= (float)pow(10, 3);
        float vy = rand() % (500) - 250.;
        vy *= (float)pow(10, 3);
        float mass = (rand() % 100 + 1) * (float)pow(10, 22);
        *(arr + i) = MassObject(x, y, vx, vy, mass, i);
    }
    for (int i = benchmark1; i < numberOfObjects; i++) {
        float x = (randfloat(0, 4)) * (float)pow(10, 10);
        float y = (randfloat(0, 4)) * (float)pow(10, 10);
        float vx = rand() % (500) - 250;
        vx *= (float)pow(10, 4);
        float vy = rand() % (500) - 250;
        vy *= (float)pow(10, 4);
        float mass = (rand() % 100 + 1) * (float)pow(10, 22);
        *(arr + i) = MassObject(x, y, vx, vy, mass, i);
    }
}

int main()
{
    srand(SEED_VALUE);
    int px = 800;
    int pz = 800;
    int numberOfObjects = 512;
    float stepsize = 25;
    std::cout << "The frame width is " << px << "." << std::endl;
    std::cout << "The frame height is " << pz << "." << std::endl;
    std::cout << "The number of objects used is " << numberOfObjects << "." << std::endl;

    //initialize objects
    MassObject** allArrs = new MassObject * [NUMBER_OF_CYCLES];
    allArrs[0] = new MassObject[numberOfObjects];
    int* remainingObjs = new int[NUMBER_OF_CYCLES];
    remainingObjs[0] = numberOfObjects;
    init(px, pz, numberOfObjects, allArrs[0]);

    std::cout << "MassObjects initialized" << std::endl;
    std::cout << "Beginning simulation... " << std::endl;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    //perform simulations
    nbodyHelperFunction(allArrs, remainingObjs, px, pz, stepsize);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;

    std::cout << "Simulation completed in " << elapsed_time.count() << " s\n";

    // write allArrs data into a text file
    std::ofstream myfile;
    myfile.open("objectsData.txt");
    std::cout << "Output data to objectsData.txt...\n";
    for (int i = 0; i < NUMBER_OF_CYCLES; i++) {
        for (int j = 0; j < remainingObjs[i]; j++) {
            myfile << allArrs[i][j].getObjNumber();
            myfile << " " << allArrs[i][j].getMass();

            myfile << " " << allArrs[i][j].getPosition_x();
            myfile << " " << allArrs[i][j].getPosition_y();

            myfile << " " << allArrs[i][j].getax();
            myfile << " " << allArrs[i][j].getay();

            myfile << " " << allArrs[i][j].getvx();
            myfile << " " << allArrs[i][j].getvy() << std::endl;
        }
    }
    myfile.close();

    // draw frames if not a dry run
    if (!DRY_RUN) {
        //initialize output buffer
        unsigned char*** buffer = new unsigned char** [pz];
        for (int i = 0; i < pz; i++)
        {
            buffer[i] = new unsigned char* [px];
            for (int j = 0; j < px; j++)
            {
                buffer[i][j] = new unsigned char[3];
            }
        }

        std::cout << "Buffer initialized and drawing frames..." << std::endl;
        for (int i = 0; i < NUMBER_OF_CYCLES; i++) {
            fill_background(buffer, px, pz, BACKGROUND_COLOR);
            for (int j = 0; j < remainingObjs[i]; j++) {
                struct r_circle thisObject;
                set_circle_values(thisObject, allArrs[i][j], px, pz);
                fill_circle(buffer, px, pz, thisObject);
            }
            //create a frame for every other cycle
            if (i % 2 == 0) {
                write_bmp_file(i / 2, buffer, px, pz);
            }
        }
        std::cout << "Output images generated." << std::endl;
        delete[] buffer;
    }

    for (int i = 0; i < NUMBER_OF_CYCLES; i++) {
        delete(allArrs[i]);
    }
    delete[] allArrs;
    delete[] remainingObjs;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t nbodyHelperFunction(MassObject** allArrs, int* remainingObjs, int px, int pz, int stepsize)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = checkCuda(cudaSetDevice(0));
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    
    // initialize device pointers
    float3* dev_accIn;
    float2* dev_accOut;
    cudaStatus = checkCuda(cudaMalloc((void**)&dev_accIn, remainingObjs[0] * sizeof(float3)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = checkCuda(cudaMalloc((void**)&dev_accOut, remainingObjs[0] * sizeof(float3)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        cudaFree(dev_accIn);
        goto Error;
    }

    for (int i = 1; i < NUMBER_OF_CYCLES; i++) {
        // cudaCopy the ax, ay, and mass from objArray to dev_accIn
        float3* accIn = (float3*)malloc(remainingObjs[i - 1] * sizeof(float3));
        for (int j = 0; j < remainingObjs[i - 1]; j++) {
            accIn[j].x = allArrs[i - 1][j].getPosition_x();
            accIn[j].y = allArrs[i - 1][j].getPosition_y();
            accIn[j].z = allArrs[i - 1][j].getMass();
        }
        
        cudaStatus = checkCuda(cudaMemcpy(dev_accIn, accIn, remainingObjs[i - 1] * sizeof(float3), cudaMemcpyHostToDevice));
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "cudaMemcpy failed!");
            checkCuda(cudaFree(dev_accIn));
            checkCuda(cudaFree(dev_accOut));
            goto Error;
        }

        dim3 threadsPerBlock(TILE_WIDTH);
        dim3 blocks(remainingObjs[i-1] / TILE_WIDTH);
        calculateNaiveAcc <<<threadsPerBlock, blocks >>>(dev_accIn, dev_accOut);

        // Check for any errors launching the kernel
        cudaStatus = checkCuda(cudaGetLastError());
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            checkCuda(cudaFree(dev_accIn));
            checkCuda(cudaFree(dev_accOut));
            goto Error;
        }

        // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
        // any errors encountered during the launch.
        cudaStatus = checkCuda(cudaDeviceSynchronize());
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            checkCuda(cudaFree(dev_accIn));
            checkCuda(cudaFree(dev_accOut));
            goto Error;
        }

        // retrieve result data from device back to host
        float2* accOut = (float2*)malloc(remainingObjs[i - 1] * sizeof(float2));
        cudaStatus = checkCuda(cudaMemcpy(accOut, dev_accOut, remainingObjs[i - 1] * sizeof(float2), cudaMemcpyDeviceToHost));
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "cudaMemcpy failed!");
            checkCuda(cudaFree(dev_accIn));
            checkCuda(cudaFree(dev_accOut));
            free(accIn);
            free(accOut);
            goto Error;
        }

        // Update allArrs with the result velocity and positions
        // init current iteration
        allArrs[i] = new MassObject[remainingObjs[i - 1]];
        for (int j = 0; j < remainingObjs[i - 1]; j++) {
            allArrs[i][j].setAcceleration(accOut[j].x, accOut[j].y);
            allArrs[i][j].changePosition(stepsize);
        }

        // Check for collisions and update arr contents
        //check if any objects have collided
        remainingObjs[i] = check_collisions(allArrs[i], remainingObjs[i - 1], px, pz);

        free(accIn);
        free(accOut);
    }

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = checkCuda(cudaDeviceReset());
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaDeviceReset failed!");
        checkCuda(cudaFree(dev_accIn));
        checkCuda(cudaFree(dev_accOut));
        goto Error;
    }
    
    Error:

    return cudaStatus;
}
