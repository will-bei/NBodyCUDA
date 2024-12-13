#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <ctime>
#include <random>
#include "HelperFunctions.h"

const unsigned int SEED_VALUE = 2024;
const bool DRY_RUN = false;

//number of cycles done for the simulation
const int NUMBEROFCYCLES = 50000;

//semi-randomly initialize the MassObjects given the field size and the number of objects
//all objects are randomly initialized with a mass between 10^22 kg to 10^24 kg
//40% of the objects will be initialized in a central 2.5*10^10 by 2.5*10^10 field
//The remaining 60% can spawn anywhere in the frame's field
void init1(int px, int pz, int numberOfObjects, MassObject* arr) {
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

int main() {
	srand(SEED_VALUE);
	int px = 800;
	int pz = 800;
	int numberOfObjects = 1024;
	float stepsize = 7200;
	std::cout << "The frame width is " << px << "." << std::endl;
	std::cout << "The frame height is " << pz << "." << std::endl;
	std::cout << "The number of objects used is " << numberOfObjects << "." << std::endl;

	//initialize objects
	MassObject** allArrs = new MassObject * [NUMBEROFCYCLES];
	allArrs[0] = new MassObject[numberOfObjects];
	int* remainingObjs = new int[NUMBEROFCYCLES];
	init2(FIELDX, FIELDY, numberOfObjects, allArrs[0]);

	std::cout << "MassObjects initialized" << std::endl;
	std::cout << "Beginning simulation... " << std::endl;
	std::chrono::time_point<std::chrono::system_clock> start, end;

	// perform simulations
	start = std::chrono::system_clock::now();
	remainingObjs[0] = numberOfObjects;
	for (int i = 1; i < NUMBEROFCYCLES; i++) {
		// init current iteration
		allArrs[i] = new MassObject[remainingObjs[i - 1]];
		for (int j = 0; j < remainingObjs[i - 1]; j++) {
			allArrs[i][j] = allArrs[i - 1][j];
		}

		//update the accelerations of the objects
		calculateAccelerations(allArrs[i], remainingObjs[i - 1]);

		//update each objects position
		for (int j = 0; j < remainingObjs[i - 1]; j++) {
			(allArrs[i][j]).changePositionFromAcc(stepsize);
		}
		
		//check if any objects have collided
		remainingObjs[i] = check_collisions(allArrs[i], remainingObjs[i - 1], px, pz);
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_time = end - start;

	std::cout << "Simulation completed in " << elapsed_time.count() << " s\n";

	// write allArrs data into a text file
	std::ofstream myfile;
	myfile.open("objectsData.txt");
	std::cout << "Output data to objectsData.txt...\n";
	/*for (int i = 0; i < NUMBEROFCYCLES; i++) {
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
	myfile.close();*/

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
		for (int i = 0; i < NUMBEROFCYCLES; i++) {
			fill_background(buffer, px, pz, BACKGROUND_COLOR);
			for (int j = 0; j < remainingObjs[i]; j++) {
				struct r_circle thisObject;
				set_circle_values(thisObject, allArrs[i][j], px, pz);
				fill_circle(buffer, px, pz, thisObject);
			}
			//create a frame for every 1000 cycle
			if (i % 4 == 0) {
				write_bmp_file(i / 4, buffer, px, pz);
			}
		}
		std::cout << "Output images generated." << std::endl;
		delete [] buffer;
	}

	for (int i = 0; i < NUMBEROFCYCLES; i++) {
		delete(allArrs[i]);
	}
	delete [] allArrs;
	delete [] remainingObjs;
	return 0;
}