#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cmath>
#include <ctime>
#include "HelperFunctions.h"

//semi-randomly initialize the MassObjects given the field size and the number of objects
//all objects are randomly initialized with a mass between 10^22 kg to 10^24 kg
//40% of the objects will be initialized in a central 2.5*10^10 by 2.5*10^10 field
//The remaining 60% can spawn anywhere in the frame's field
void init(int px, int pz, int numberOfObjects, MassObject* arr) {
	int benchmark1 = numberOfObjects * 4 / 10;
	for (int i = 0; i < benchmark1; i++) {
		float x = (0.5 + randfloat(0, 2.5)) * pow(10, 10);
		float y = (0.5 + randfloat(0, 2.5)) * pow(10, 10);
		float vx = rand() % (500) - 250;
		vx *= pow(10, 3);
		float vy = rand() % (500) - 250;
		vy *= pow(10, 3);
		float mass = (rand() % 100 + 1) * pow(10, 22);
		*(arr + i) = MassObject(x, y, vx, vy, mass, i);
	}
	for (int i = benchmark1; i < numberOfObjects; i++) {
		float x = (randfloat(0, 4)) * pow(10, 10);
		float y = (randfloat(0, 4)) * pow(10, 10);
		float vx = rand() % (500) - 250;
		vx *= pow(10, 4);
		float vy = rand() % (500) - 250;
		vy *= pow(10, 4);
		float mass = (rand() % 100 + 1) * pow(10, 22);
		*(arr + i) = MassObject(x, y, vx, vy, mass, i);
	}
}

int main() {
	srand(time(0));
	int px;
	int pz;
	int numberOfObjects;
	float stepsize = 25;
	px = 800;
	pz = 800;
	numberOfObjects = 300;
	std::cout << "The frame width is " << px << "." << std::endl;
	std::cout << "The frame height is " << pz << "." << std::endl;
	std::cout << "The number of objects used is " << numberOfObjects << "." << std::endl;

	//initialize output buffer
	unsigned char*** buffer = new unsigned char** [pz];
	int i, j;
	for (i = 0; i < pz; i++)
	{
		buffer[i] = new unsigned char* [px];
		for (j = 0; j < px; j++)
		{
			buffer[i][j] = new unsigned char[3];
		}
	}
	std::cout << "Buffer initialized." << std::endl;

	//initialize objects
	MassObject** allArrs = new MassObject * [NUMBEROFCYCLES];
	allArrs[0] = new MassObject[numberOfObjects];
	int* remainingObjs = new int[NUMBEROFCYCLES];
	init(px, pz, numberOfObjects, allArrs[0]);

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
			(allArrs[i][j]).changePosition(stepsize);
		}
		
		//check if any objects have collided
		remainingObjs[i] = check_collisions(allArrs[i], remainingObjs[i - 1], px, pz);
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_time = end - start;

	std::cout << "Simulation completed in " << elapsed_time.count() << " ms\n";

	// draw frames

	for (int i = 0; i < NUMBEROFCYCLES; i++) {
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

	return 0;
}