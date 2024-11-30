#pragma once
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>

//struct for color
struct p_color
{
	unsigned char r; //red, 0 to 255
	unsigned char g; //green, 0 to 255
	unsigned char b; //blue, 0 to 255
};

//struct for circle
struct r_circle
{
	float cx; //x-coordinate of center of circle
	float cy; //y-coordinate of center of circle
	int radius; //pixel radius of the circle
	struct p_color circle_color; //color of the circle
};

//gravitational constant
const float GRAV_CONST = 6.67 / (float)pow(10, 11);
//simulation field size
const float FIELDX = 4 * (float)pow(10, 10);
const float FIELDY = 4 * (float)pow(10, 10);
const p_color BACKGROUND_COLOR = {
	0, // R
	0, // G
	5, // B
};

//class MassObject
class MassObject {
private:
	int objnumber; // the position in which the MassObject was initially initialized
	//primarily used for debugging
	float mass; // the mass of the object

	//position
	float x0; // initial position x
	float y0; // initial position y
	float x; //current position x
	float y; //current position y

	//velocity
	float vx; //x-component
	float vy; //y-component

	//acceleration
	float ax; //x-component
	float ay; //y-component
public:
	MassObject() {}; //default constructor
	MassObject(float, float, float, float, float, int); //initializes the massobject at a random x,y on the field and mass from 1-10
	float getMass() const; //returns the mass of the object
	float getax() const; // returns the x-component of the acceleration of the object
	float getay() const; // returns the y-component of the acceleration of the object
	float getvx() const; // returns the x-component of the velocity of the object
	float getvy() const; // returns the y-component of the velocity of the object
	float getPosition_x() const; // returns the object's x-coordinate
	float getPosition_y() const; // returns the object's y-coordinate
	int getObjNumber() const; // returns the objnumber of the MassObject - see above
	void changeMass(float delta_mass); //increases the mass of the object by delta_mass
	void setAcceleration(float ax_new, float ay_new); //set the x- and y-components of acceleration to the respective parameter values
	void changeAcceleration(float d_ax, float d_ay); //change the accelerations by d_ax and d_ay
	void changeV(float vx_new, float vy_new); //sets the velocities of the object to the respective parameter values
	void changePosition(float stepsize); //changes the position of the object given a stepsize (i.e. change in time)
};
MassObject::MassObject(float start_x, float start_y, float start_vx, float start_vy, float start_mass, int ob_no) {
	x0 = start_x;
	y0 = start_y;
	x = x0;
	y = y0;
	ax = 0;
	ay = 0;
	vx = start_vx;
	vy = start_vy;
	mass = start_mass;
	objnumber = ob_no;
}
float MassObject::getMass() const {
	return mass;
}
float MassObject::getax() const {
	return ax;
}
float MassObject::getay() const {
	return ay;
}
float MassObject::getvx() const {
	return vx;
}
float MassObject::getvy() const {
	return vy;
}
float MassObject::getPosition_x() const {
	return x;
}
float MassObject::getPosition_y() const {
	return y;
}
int MassObject::getObjNumber() const {
	return objnumber;
}
void MassObject::changeMass(float delta_mass) {
	mass += delta_mass;
}
void MassObject::setAcceleration(float ax_new, float ay_new) {
	ax = ax_new;
	ay = ay_new;
}
void MassObject::changeAcceleration(float d_ax, float d_ay) {
	ax += d_ax;
	ay += d_ay;
}
void MassObject::changeV(float vx_new, float vy_new) {
	vx = vx_new;
	vy = vy_new;
}
void MassObject::changePosition(float stepsize) {
	vx += ax * stepsize;
	vy += ay * stepsize;

	x += 0.5 * ax * stepsize * stepsize + stepsize * vx;
	y += 0.5 * ay * stepsize * stepsize + stepsize * vy;
}

//converts int to a five digit string
std::string int_to_five_digit_string(int frame_number)
{
	std::ostringstream strm;
	strm << std::setfill('0') << std::setw(5) << frame_number;
	return strm.str();
}
//converts string to int
int string_to_int(std::string s)
{
	std::istringstream strm;
	strm.str(s);
	int n = 0;
	strm >> n;
	return n;
}

//write a bmp header file
void write_bmp_header_file(std::ofstream& output_file, int px, int pz)
{
	unsigned short int bfType;
	bfType = 0x4D42;
	output_file.write((char*)&bfType, sizeof(short int));

	unsigned int bfSize;
	int rem;
	rem = 3 * px % 4;
	int padding;
	if (rem == 0)
	{
		padding = 0;
	}
	else
	{
		padding = 4 - rem;
	}

	bfSize = 14 + 40 + (3 * px + padding) * pz;
	//	bfSize = 14 + 40 + (3 * px+padding) * pz + 2;
	output_file.write((char*)&bfSize, sizeof(int));

	unsigned short int bfReserved1;
	bfReserved1 = 0;
	output_file.write((char*)&bfReserved1, sizeof(short int));

	unsigned short int bfReserved2;
	bfReserved2 = 0;
	output_file.write((char*)&bfReserved2, sizeof(short int));

	unsigned int bfOffsetBits;
	bfOffsetBits = 14 + 40;
	output_file.write((char*)&bfOffsetBits, sizeof(int));

	unsigned int biSize;
	biSize = 40;
	output_file.write((char*)&biSize, sizeof(int));

	int biWidth;
	biWidth = px;
	output_file.write((char*)&biWidth, sizeof(int));

	int biHeight;
	biHeight = pz;
	output_file.write((char*)&biHeight, sizeof(int));

	unsigned short int biPlanes;
	biPlanes = 1;
	output_file.write((char*)&biPlanes, sizeof(short int));

	unsigned short int biBitCount;
	biBitCount = 24;
	output_file.write((char*)&biBitCount, sizeof(short int));

	unsigned int biCompression;
	// #define BI_RGB 0
	unsigned int bi_rgb = 0;
	//	biCompression=BI_RGB;
	biCompression = bi_rgb;
	output_file.write((char*)&biCompression, sizeof(int));

	unsigned int biSizeImage;
	biSizeImage = 0;
	output_file.write((char*)&biSizeImage, sizeof(int));

	unsigned int biXPelsPerMeter;
	biXPelsPerMeter = 0;
	output_file.write((char*)&biXPelsPerMeter, sizeof(int));

	unsigned int biYPelsPerMeter;
	biYPelsPerMeter = 0;
	output_file.write((char*)&biYPelsPerMeter, sizeof(int));

	unsigned int biClrUsed;
	biClrUsed = 0;
	output_file.write((char*)&biClrUsed, sizeof(int));

	unsigned int biClrImportant;
	biClrImportant = 0;
	output_file.write((char*)&biClrImportant, sizeof(int));
}

//write a bmp file
void write_bmp_file(int f_number, unsigned char*** output_buffer, int px, int pz)
{
	std::ofstream ostrm_1;
	std::string o_file_name = int_to_five_digit_string(f_number) + ".bmp";
	ostrm_1.open(o_file_name.c_str(), std::ios::out | std::ios::binary);
	if (ostrm_1.fail())
	{
		std::cout << "Error.  Can't open output file " << o_file_name << "." << std::endl;
		return;
	}
	// std::cout << "Opening output file " << o_file_name << "." << std::endl;

	int rem;
	rem = 3 * px % 4;
	int padding;
	if (rem == 0)
	{
		padding = 0;
	}
	else
	{
		padding = 4 - rem;
	}
	//cout << "padding is " << padding << "." << endl;
	//cout << "rem is "  << rem << "." << endl;
	write_bmp_header_file(ostrm_1, px, pz);

	unsigned char p_buffer[4];
	p_buffer[0] = 0;
	p_buffer[1] = 0;
	p_buffer[2] = 0;
	p_buffer[3] = 0;

	unsigned char* line_buffer = new unsigned char[px * 3];

	int i;
	int j;
	for (i = pz - 1; i >= 0; i--)
	{
		for (j = 0; j < px; j++)
		{
			line_buffer[3 * j + 0] = output_buffer[i][j][2];
			line_buffer[3 * j + 1] = output_buffer[i][j][1];
			line_buffer[3 * j + 2] = output_buffer[i][j][0];
		}
		ostrm_1.write((char*)line_buffer, px * 3 * sizeof(unsigned char));
		ostrm_1.write((char*)p_buffer, padding * sizeof(unsigned char));
	}
	delete[] line_buffer;
	line_buffer = NULL;
	ostrm_1.close();
}

//fills the background of the frame
void fill_background(unsigned char*** o_buffer, int px, int pz, p_color bg_color)
{
	int i;
	int j;
	for (i = 0; i < pz; i++)
	{
		for (j = 0; j < px; j++)
		{
			o_buffer[i][j][0] = bg_color.r;
			o_buffer[i][j][1] = bg_color.g;
			o_buffer[i][j][2] = bg_color.b;
		}
	}
}

//draws a circle (representative of a MassObject)
void fill_circle(unsigned char*** o_buffer, int px, int py, r_circle s_circle)
{
	// calculate for loop bounds
	int leftBound = s_circle.cx - 2 * s_circle.radius > 0 ? s_circle.cx - 2 * s_circle.radius : 0;
	int rightBound = s_circle.cx + 2 * s_circle.radius < px ? s_circle.cx + 2 * s_circle.radius : px;
	int topBound = s_circle.cy - 2 * s_circle.radius > 0 ? s_circle.cy - 2 * s_circle.radius : 0;
	int bottomBound = s_circle.cy + 2 * s_circle.radius < py ? s_circle.cy + 2 * s_circle.radius : py;

	for (int i = topBound; i < bottomBound; i++) {
		for (int j = leftBound; j < rightBound; j++) {
			float d = 
				sqrt((float)(s_circle.cx - j) * (float)(s_circle.cx - j) + (float)(s_circle.cy - i) * (float)(s_circle.cy - i));
			if (d <= s_circle.radius) {
				o_buffer[i][j][0] = s_circle.circle_color.r;
				o_buffer[i][j][1] = s_circle.circle_color.g;
				o_buffer[i][j][2] = s_circle.circle_color.b;
			}
		}
	}
}

//sets circle values based on the Mass of a MassObject
//redder objects are lighter and smaller
//yellower objects are heavier and larger
void set_circle_values(r_circle& thisObject, MassObject mo, int px, int pz)
{
	float value = log10(mo.getMass() / (float)pow(10, 22));
	thisObject.circle_color.r = (int)(255 - 95 * pow(0.804, value));
	thisObject.circle_color.g = (int)(255 - 255 * pow(0.725, value));
	thisObject.circle_color.b = 0;
	thisObject.cx = mo.getPosition_x() * px / FIELDX;
	thisObject.cy = mo.getPosition_y() * pz / FIELDY;
	thisObject.radius = (int)(sqrt(mo.getMass()) / (float)pow(10, 11) / 5. + 0.5);
	if (thisObject.radius < 1) {
		thisObject.radius = 1;
	}
}

//the next two methods recursively sort MassObjects in order of decreasing mass
//sorting algorithm for this: quicksort
int partition(MassObject* A, int p, int q)
{
	MassObject x = *(A + p);
	int i = p;
	int j;

	for (j = p + 1; j < q; j++)
	{
		if ((*(A + j)).getMass() >= x.getMass())
		{
			i = i + 1;
			MassObject temp = *(A + j);
			*(A + j) = *(A + i);
			*(A + i) = temp;
		}

	}

	MassObject temp = *(A + p);
	*(A + p) = *(A + i);
	*(A + i) = temp;
	return i;
}
void sort_MassObjects(MassObject* A, int p, int q) {
	int r;
	if (p < q)
	{
		r = partition(A, p, q);
		sort_MassObjects(A, p, r);
		sort_MassObjects(A, r + 1, q);
	}
}


//checks if any collisions have occurred in this time frame
//if collisions have occurred, the smaller of the "destroyed" objects is moved to the end
//of the list and the size is reduced by one
bool isInRange(MassObject A, MassObject B, int px, int pz) {
	float dx = abs((A.getPosition_x() - B.getPosition_x())) * px / FIELDX;
	float dy = abs((A.getPosition_y() - B.getPosition_y())) * pz / FIELDY;
	float dr = sqrt(dx * dx + dy * dy);
	bool result = (dr < sqrt(A.getMass()) / pow(10, 11) / 5 + 0.5);

	return (result);
}

//given object A and object B that are colliding in a perfectly
//inelastic collision, finds the resulting object and velocity and
//stores the values in object A
MassObject* updateObjects(MassObject* A, MassObject* B) {
	float m1 = (*A).getMass();
	float m2 = (*B).getMass();
	float v1 = (*A).getvx();
	float v2 = (*B).getvx();
	float vx = (m1 * v1 + m2 * v2) / (m1 + m2);

	float v3 = (*A).getvy();
	float v4 = (*B).getvy();
	float vy = (m1 * v3 + m2 * v4) / (m1 + m2);
	(*A).changeV(vx, vy);
	(*A).changeMass((*B).getMass());

	return A;
}
//moves an object in the MassObject* A array to the back
//of the array, given the object's position in the array

void swapObjects(MassObject* A, int position, int size) {
	MassObject temp = *(A + position);
	for (int k = position; k < size - 1; k++) {
		*(A + k) = *(A + k + 1);
	}
	*(A + size - 1) = temp;
}

//check to see if any collisions occured in an array of MassObjects
int check_collisions(MassObject* A, int size, int px, int pz) {
	for (int i = 0; i < size; i++) {
		for (int j = i + 1; j < size; j++) {
			MassObject* object1 = (A + i);
			MassObject* object2 = (A + j);
			if (isInRange(*object1, *object2, px, pz)) {
				// (m1*vx1+m2*vx2)/(m1+m2)
				object1 = updateObjects(object1, object2);
				//swap smaller object with last object, and shorten array
				swapObjects(A, j, size);
				size--;
			}
		}
	}
	return size;
}

//find the dx between two objects
float find_dx(MassObject A, MassObject B) {
	return (B.getPosition_x() - A.getPosition_x());
}
//find the dy between two objects
float find_dy(MassObject A, MassObject B) {
	return (B.getPosition_y() - A.getPosition_y());
}
//find the straight-line distance given dx and dy
float find_rsqrd(float dx, float dy) {
	float rsqrd = dx * dx + dy * dy;

	return rsqrd;
}
//find the acceleration by an object given that object and the distance
float set_acc(float rsqrd, MassObject A) {
	if (rsqrd == 0) {
		return 0;
	}

	float acc;
	acc = 6.67 * (A.getMass()) / rsqrd / 10000;

	return acc;
}
//finds the components of the acceleration given dx dy
void find_components(MassObject* A, float net_acc, float dx, float dy) {
	float theta = atan2(dy, dx);
	float ax = net_acc * cos(theta);
	float ay = net_acc * sin(theta);
	(*A).changeAcceleration(ax, ay);
}
//calculates the accelerations of all the MassObjects in the field
//The net acceleration of an object is calculated by adding the accelerations
//on that object from all other objects in the array.
void calculateAccelerations(MassObject* A, int size) {
	for (int i = 0; i < size; i++) {
		(*(A + i)).setAcceleration(0, 0);
		for (int j = 0; j < size; j++) {
			if (i != j) {
				MassObject* object1 = (A + i);
				MassObject* object2 = (A + j);
				float dx = find_dx(*object1, *object2);
				float dy = find_dy(*object1, *object2);
				float rsqrd = find_rsqrd(dx, dy);
				float net_acc = set_acc(rsqrd, *object2);
				find_components(object1, net_acc, dx, dy);
			}
		}
	}
}
//generate a random decimal between fMin and fMax
float randfloat(float fMin, float fMax) {
	float f = ((float)rand() / (RAND_MAX));
	float i = (float)(rand() % (int)(fMax - fMin));
	float result = f + i;

	if (result < 0) {
		result *= -1;
	}
	return result;
}
