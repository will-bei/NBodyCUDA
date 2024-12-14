
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include "HelperFunctions.h"
#include "ErrorChecker.cuh"


#define NUMBER_OF_CYCLES 1000
#define CYCLES_PER_IMAGE 2
#define TILE_WIDTH 256

const unsigned int SEED_VALUE = 2024;
const bool DRY_RUN = false;

cudaError_t nbodyHelperFunction(MassObject** allArrs, int* remainingObjs, int px, int py, int stepsize, double& calculationTime);

__device__ float CUDA_GRAV_CONST = 6.67e-11;

__global__ void calculateCorseAcc(float3* pos, float2* vel, float2* globalV, float2* globalPos, int stepsize, int size) {
    //x and y are the 2D positions and z is the weight of the particle  
    int i, j, tile, c;
    float2 acc = { 0.0f, 0.0f };
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int COARSENING_FACTOR = 1; //HAS to be less than TILE_WIDTH
    globalV[gtid] = vel[gtid];
    globalPos[gtid].x = pos[gtid].x;
    globalPos[gtid].y = pos[gtid].y;
    //acceleration calculation
    for (c = 0; c < COARSENING_FACTOR; c++) //haha c++
    {
        acc = { 0.0f, 0.0f };
        float2 currentPos = globalPos[gtid];
        float2 currentV = globalV[gtid];
        __syncthreads();

        for (j = 0; j < blockDim.x; j++) {
            float2 vec;

            //vector from current particle to its computational partner particle
            vec.x = currentPos.x - globalPos[j].x;
            vec.y = currentPos.y - globalPos[j].y;
            //distance squared calculation
            float sqrddist = vec.x * vec.x + vec.y * vec.y;

            if (sqrddist > 0) {
                //net_acc  from this object
                float net_acc = -CUDA_GRAV_CONST * pos[j].z / sqrddist;

                //increment acceleration
                acc.x += cosf(atan2f(vec.y, vec.x)) * net_acc;
                acc.y += sinf(atan2f(vec.y, vec.x)) * net_acc;
            }
        }
    
        // update velocity and position from this
        currentV.x += acc.x*stepsize;
        currentV.y += acc.y*stepsize;

        currentPos.x += 0.5 * acc.x * stepsize * stepsize + stepsize * currentV.x;
        currentPos.y += 0.5 * acc.y * stepsize * stepsize + stepsize * currentV.y;
        __syncthreads();

        globalV[gtid].x = currentV.x;
        globalV[gtid].y = currentV.y;
        globalPos[gtid].x = currentPos.x;
        globalPos[gtid].y = currentPos.y;
    }
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

//initialize objects in a normal distribution, centered on fieldX/2 and fieldY/2
void init2(float fieldX, float fieldY, int numberOfObjects, MassObject* arr) {
    std::default_random_engine generator;
    generator.seed(SEED_VALUE);
    std::normal_distribution<float> distributionX(fieldX / 2, fieldX / 4);
    std::normal_distribution<float> distributionY(fieldY / 2, fieldY / 4);
    std::normal_distribution<float> distributionV(0, 500);

    for (int i = 0; i < numberOfObjects; i++) {
        float x = distributionX(generator);
        float y = distributionY(generator);
        float vx = distributionV(generator);
        float vy = distributionV(generator);
        float mass = (rand() % 100 + 1) * (float)pow(10, 22);
        *(arr + i) = MassObject(x, y, vx, vy, mass, i);
    }
}

//initialize objects in a three normal distribution, with different avg. initial velocities
void init3(float fieldX, float fieldY, int numberOfObjects, MassObject* arr) {
    std::default_random_engine generator;
    generator.seed(SEED_VALUE);
    std::normal_distribution<float> distributionXl(fieldX / 3, fieldX / 8);
    std::normal_distribution<float> distributionYl(fieldY / 3, fieldY / 8);
    std::normal_distribution<float> distributionV1(-400, 50);
    std::normal_distribution<float> distributionV2(400, 50);

    for (int i = 0; i < numberOfObjects / 3; i++) {
        float x = distributionXl(generator);
        float y = distributionYl(generator);
        float vx = distributionV2(generator);
        float vy = distributionV1(generator);
        float mass = (rand() % 100 + 1) * (float)pow(10, 21);
        *(arr + i) = MassObject(x, y, vx, vy, mass, i);
    }

    std::normal_distribution<float> distributionXb(fieldX / 2, fieldX / 6);
    std::normal_distribution<float> distributionYb(3 * fieldY / 4, fieldY / 6);

    for (int i = numberOfObjects / 3; i < 2 * numberOfObjects / 3; i++) {
        float x = distributionXb(generator);
        float y = distributionYb(generator);
        float vx = distributionV2(generator);
        float vy = distributionV2(generator);
        float mass = (rand() % 100 + 1) * (float)pow(10, 20);
        *(arr + i) = MassObject(x, y, vx, vy, mass, i);
    }

    std::normal_distribution<float> distributionXr(2 * fieldX / 3, 2 * fieldX / 8);
    std::normal_distribution<float> distributionYr(fieldY / 2, fieldY / 8);

    for (int i = 2 * numberOfObjects / 3; i < numberOfObjects; i++) {
        float x = distributionXr(generator);
        float y = distributionYr(generator);
        float vx = distributionV1(generator);
        float vy = distributionV2(generator);
        float mass = (rand() % 100 + 1) * (float)pow(10, 20);
        *(arr + i) = MassObject(x, y, vx, vy, mass, i);
    }
}

int main()
{
    srand(SEED_VALUE);
    int px = 800;
    int pz = 800;
    int numberOfObjects = 1024;
    float stepsize = 7200;
    std::cout << "The frame width is " << px << "." << std::endl;
    std::cout << "The frame height is " << pz << "." << std::endl;
    std::cout << "The number of objects used is " << numberOfObjects << "." << std::endl;

    //initialize objects
    MassObject** allArrs = new MassObject * [NUMBER_OF_CYCLES];
    allArrs[0] = new MassObject[numberOfObjects];
    int* remainingObjs = new int[NUMBER_OF_CYCLES];
    remainingObjs[0] = numberOfObjects;
    init3(FIELDX, FIELDY, numberOfObjects, allArrs[0]);

    std::cout << "MassObjects initialized" << std::endl;
    std::cout << "Beginning simulation... " << std::endl;
    double calculationTime = 0;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    //perform simulations
    nbodyHelperFunction(allArrs, remainingObjs, px, pz, stepsize, calculationTime);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;

    std::cout << "Simulation completed in " << elapsed_time.count() << " s\n";
    std::cout << "Time spent simulating: " << calculationTime << " s\n";

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
        for (int i = 0; i < NUMBER_OF_CYCLES; i += CYCLES_PER_IMAGE) {
            fill_background(buffer, px, pz, BACKGROUND_COLOR);
            for (int j = 0; j < remainingObjs[i]; j++) {
                struct r_circle thisObject;
                set_circle_values(thisObject, allArrs[i][j], px, pz);
                fill_circle(buffer, px, pz, thisObject);
            }
            // write a new img every CYCLES_PER_IMAGE
            write_bmp_file(i / CYCLES_PER_IMAGE, buffer, px, pz);
        }
        std::cout << "Output images generated." << std::endl;
        delete[] buffer;

        std::string date = current_dateTime();
        std::replace(date.begin(), date.end(), '/', '_');
        for (int i = 0; i < date.length(); i++) {
            if (date.at(i) == ':') date.erase(i, 1);
        }
        std::cout << '\"' << date << '\"' << std::endl;
        std::string cmd = "ffmpeg -framerate 50 -i outputimgs/%07d.bmp -c:v libx264 -r 50 cpuOut" + date + ".mp4";
        std::cout << '\"' << cmd << '\"' << std::endl;
        int n = cmd.length();
        char* cmdArr = new char[n];
        for (int i = 0; i < n; i++) {
            cmdArr[i] = cmd.at(i);
        }
        system(cmdArr);
        delete[] cmdArr;
    }

    for (int i = 0; i < NUMBER_OF_CYCLES; i++) {
        delete(allArrs[i]);
    }
    delete[] allArrs;
    delete[] remainingObjs;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t nbodyHelperFunction(MassObject** allArrs, int* remainingObjs, int px, int pz, int stepsize, double& calculationTime)
{
    cudaError_t cudaStatus;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> sumTime = std::chrono::seconds::zero();

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = checkCuda(cudaSetDevice(0));
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    
    // initialize device pointers
    float3* dev_accIn;
    float2* dev_velIn;
    float2* dev_velOut;
    float2* dev_posOut;

    cudaStatus = checkCuda(cudaMalloc((void**)&dev_accIn, remainingObjs[0] * sizeof(float3)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = checkCuda(cudaMalloc((void**)&dev_velIn, remainingObjs[0] * sizeof(float2)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        cudaFree(dev_accIn);
        goto Error;
    }

    cudaStatus = checkCuda(cudaMalloc((void**)&dev_velOut, remainingObjs[0] * sizeof(float2)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        cudaFree(dev_accIn);
        cudaFree(dev_velIn);
        goto Error;
    }

    cudaStatus = checkCuda(cudaMalloc((void**)&dev_posOut, remainingObjs[0] * sizeof(float2)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        cudaFree(dev_accIn);
        cudaFree(dev_velIn);
        cudaFree(dev_posOut);
        goto Error;
    }

    for (int i = 1; i < NUMBER_OF_CYCLES; i++) {
        // cudaCopy the ax, ay, and mass from objArray to dev_accIn; and velocities to dev_velIn
        float3* accIn = (float3*)malloc(remainingObjs[i - 1] * sizeof(float3));
        float2* velIn = (float2*)malloc(remainingObjs[i - 1] * sizeof(float2));
        for (int j = 0; j < remainingObjs[i - 1]; j++) {
            accIn[j].x = allArrs[i - 1][j].getPosition_x();
            accIn[j].y = allArrs[i - 1][j].getPosition_y();
            accIn[j].z = allArrs[i - 1][j].getMass();
            velIn[j].x = allArrs[i - 1][j].getvx();
            velIn[j].y = allArrs[i - 1][j].getvy();
        }
        
        cudaStatus = checkCuda(cudaMemcpy(dev_accIn, accIn, remainingObjs[i - 1] * sizeof(float3), cudaMemcpyHostToDevice));
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "cudaMemcpy failed!");
            cudaFree(dev_accIn);
            cudaFree(dev_velIn);
            cudaFree(dev_posOut);
            cudaFree(dev_velOut);
            goto Error;
        }

        cudaStatus = checkCuda(cudaMemcpy(dev_velIn, velIn, remainingObjs[i - 1] * sizeof(float2), cudaMemcpyHostToDevice));
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "cudaMemcpy failed!");
            cudaFree(dev_accIn);
            cudaFree(dev_velIn);
            cudaFree(dev_posOut);
            cudaFree(dev_velOut);
            goto Error;
        }

        dim3 threadsPerBlock(TILE_WIDTH);
        dim3 blocks(1 + remainingObjs[i - 1] / TILE_WIDTH);
        start = std::chrono::system_clock::now();
        // calculateCorseAcc(float3* pos, float2* vel, float2* globalAcc, float2* globalV, float2* globalPos, int stepsize, int size)
        calculateCorseAcc<<<threadsPerBlock, blocks>>>(dev_accIn, dev_velIn, dev_velOut, dev_posOut, stepsize, remainingObjs[i - 1]);
        end = std::chrono::system_clock::now();
        sumTime += end - start;

        // Check for any errors launching the kernel
        cudaStatus = checkCuda(cudaGetLastError());
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            cudaFree(dev_accIn);
            cudaFree(dev_velIn);
            cudaFree(dev_posOut);
            cudaFree(dev_velOut);
            goto Error;
        }

        // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
        // any errors encountered during the launch.
        cudaStatus = checkCuda(cudaDeviceSynchronize());
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            cudaFree(dev_accIn);
            cudaFree(dev_velIn);
            cudaFree(dev_posOut);
            cudaFree(dev_velOut);
            goto Error;
        }

        // retrieve result data from device back to host
        float2* vOut = (float2*)malloc(remainingObjs[i - 1] * sizeof(float2));
        cudaStatus = checkCuda(cudaMemcpy(vOut, dev_velOut, remainingObjs[i - 1] * sizeof(float2), cudaMemcpyDeviceToHost));
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "cudaMemcpy failed!");
            cudaFree(dev_accIn);
            cudaFree(dev_velIn);
            cudaFree(dev_posOut);
            cudaFree(dev_velOut);
            free(accIn);
            goto Error;
        }

        float2* posOut = (float2*)malloc(remainingObjs[i - 1] * sizeof(float2));
        cudaStatus = checkCuda(cudaMemcpy(posOut, dev_posOut, remainingObjs[i - 1] * sizeof(float2), cudaMemcpyDeviceToHost));
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "cudaMemcpy failed!");
            cudaFree(dev_accIn);
            cudaFree(dev_velIn);
            cudaFree(dev_posOut);
            cudaFree(dev_velOut);
            free(accIn);
            free(vOut);
            goto Error;
        }

        // Update allArrs with the result velocity and positions
        // init current iteration
        allArrs[i] = new MassObject[remainingObjs[i - 1]];
        for (int j = 0; j < remainingObjs[i - 1]; j++) {
            MassObject currentObj = allArrs[i - 1][j];
            currentObj.changeV(vOut[j].x, vOut[j].y);
            currentObj.setPosition(posOut[j].x, posOut[j].y);
            allArrs[i][j] = currentObj;
        }

        // Check for collisions and update arr contents
        //check if any objects have collided
        remainingObjs[i] = check_collisions(allArrs[i], remainingObjs[i - 1], px, pz);

        free(accIn);
        free(posOut);
    }

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = checkCuda(cudaDeviceReset());
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaDeviceReset failed!");
        checkCuda(cudaFree(dev_accIn));
        checkCuda(cudaFree(dev_posOut));
        goto Error;
    }
    
    Error:
    calculationTime = sumTime.count();

    return cudaStatus;
}
