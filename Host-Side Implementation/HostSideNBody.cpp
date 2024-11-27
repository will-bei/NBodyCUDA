#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>

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
	double cx; //x-coordinate of center of circle
	double cy; //y-coordinate of center of circle
	double radius; //radius of the circle
	struct p_color circle_color; //color of the circle
};

using namespace std;

//gravitational constant
const double GRAV_CONST = 6.67 / pow(10, 11);
//number of cycles done for the simulation
const int NUMBEROFCYCLES = 4000;
//simulation field size
const double FIELDX = 4 * pow(10, 10);
const double FIELDY = 4 * pow(10, 10);

//class MassObject
//a MassObject object 
class MassObject {
private:
	int objnumber; // the position in which the MassObject was initially initialized
	//primarily used for debugging
	double mass; // the mass of the object

	//position
	double x0; // initial position x
	double y0; // initial position y
	double x; //current position x
	double y; //current position y

	//velocity
	double vx; //x-component
	double vy; //y-component

	//acceleration
	double ax; //x-component
	double ay; //y-component
public:
	MassObject() {}; //default constructor
	MassObject(double, double, double, double, double, int); //initializes the massobject at a random x,y on the field and mass from 1-10
	double getMass() const; //returns the mass of the object
	double getax() const; // returns the x-component of the acceleration of the object
	double getay() const; // returns the y-component of the acceleration of the object
	double getvx() const; // returns the x-component of the velocity of the object
	double getvy() const; // returns the y-component of the velocity of the object
	double getPosition_x() const; // returns the object's x-coordinate
	double getPosition_y() const; // returns the object's y-coordinate
	int getObjNumber() const; // returns the objnumber of the MassObject - see above
	void changeMass(double delta_mass); //increases the mass of the object by delta_mass
	void setAcceleration(double ax_new, double ay_new); //set the x- and y-components of acceleration to the respective parameter values
	void changeAcceleration(double d_ax, double d_ay); //change the accelerations by d_ax and d_ay
	void changeV(double vx_new, double vy_new); //sets the velocities of the object to the respective parameter values
	void changePosition(double stepsize); //changes the position of the object given a stepsize (i.e. change in time)
};
MassObject::MassObject(double start_x, double start_y, double start_vx, double start_vy, double start_mass, int ob_no) {
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
double MassObject::getMass() const {
	return mass;
}
double MassObject::getax() const {
	return ax;
}
double MassObject::getay() const {
	return ay;
}
double MassObject::getvx() const {
	return vx;
}
double MassObject::getvy() const {
	return vy;
}
double MassObject::getPosition_x() const {
	return x;
}
double MassObject::getPosition_y() const {
	return y;
}
int MassObject::getObjNumber() const {
	return objnumber;
}
void MassObject::changeMass(double delta_mass) {
	mass += delta_mass;
}
void MassObject::setAcceleration(double ax_new, double ay_new) {
	ax = ax_new;
	ay = ay_new;
}
void MassObject::changeAcceleration(double d_ax, double d_ay) {
	ax += d_ax;
	ay += d_ay;
}
void MassObject::changeV(double vx_new, double vy_new) {
	vx = vx_new;
	vy = vy_new;
}
void MassObject::changePosition(double stepsize) {
	vx += ax * stepsize;
	vy += ay * stepsize;
	//due to small stepsize, 0.5*a*t*t is negigible
	x += stepsize * vx;
	y += stepsize * vy;
}

//converts int to a five digit string
string int_to_five_digit_string(int frame_number)
{
	ostringstream strm;
	strm << setfill('0') << setw(5) << frame_number;
	return strm.str();
}
//converts string to int
int string_to_int(string s)
{
	istringstream strm;
	strm.str(s);
	int n = 0;
	strm >> n;
	return n;
}

//write a bmp header file
//method recycled from animation_project
void write_bmp_header_file(ofstream& output_file, int px, int pz)
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
//method recycled from animation_project
void write_bmp_file(int f_number, string output_file_name, unsigned char*** output_buffer, int px, int pz)
{
	ofstream ostrm_1;
	//string o_file_name = int_to_five_digit_string(f_number)+output_file_name;
	string o_file_name = int_to_five_digit_string(f_number) + ".bmp";
	ostrm_1.open(o_file_name.c_str(), ios::out | ios::binary);
	if (ostrm_1.fail())
	{
		cout << "Error.  Can't open output file " << o_file_name << "." << endl;
		return;
	}
	cout << "Opening output file " << o_file_name << "." << endl;

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
//method recycled from animation_project
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
//method recycled from animation_project
void fill_circle(unsigned char*** o_buffer, int px, int pz, r_circle s_circle)
{
	//cout << s_circle.radius << ": ";
	//cout << s_circle.cx << " " << s_circle.cy;
	for (int i = 0; i < pz; i++) {
		for (int j = 0; j < px; j++) {
			double d = sqrt((s_circle.cx - j) * (s_circle.cx - j) + (s_circle.cy - i) * (s_circle.cy - i));
			if (d <= s_circle.radius) {
				//cout << "Painted.";
				o_buffer[i][j][0] = s_circle.circle_color.r;
				o_buffer[i][j][1] = s_circle.circle_color.g;
				o_buffer[i][j][2] = s_circle.circle_color.b;
			}
		}
	}
	//cout << endl << endl;
}

//sets circle values based on the Mass of a MassObject
//redder objects are lighter and smaller
//yellower objects are heavier and larger
void set_circle_values(r_circle& thisObject, MassObject mo, int px, int pz)
{
	double value = log10(mo.getMass() / pow(10, 22));
	thisObject.circle_color.r = (int)(255 - 95 * pow(0.804, value));
	thisObject.circle_color.g = (int)(255 - 255 * pow(0.725, value));
	thisObject.circle_color.b = 0;
	thisObject.cx = mo.getPosition_x() * px / FIELDX;
	thisObject.cy = mo.getPosition_y() * pz / FIELDY;
	//cout << mo.getPosition_x() << " " << mo.getPosition_y() << " ";
	//cout << thisObject.cx << " " << thisObject.cy << endl;
	thisObject.radius = (int)(sqrt(mo.getMass()) / pow(10, 11) / 5 + 0.5);
	if (thisObject.radius == 0) {
		thisObject.radius++;
	}
	//cout << "IF: " << thisObject.cx << "  " << thisObject.cy << "  " << thisObject.radius << endl;
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
	double dx = abs((A.getPosition_x() - B.getPosition_x())) * px / FIELDX;
	double dy = abs((A.getPosition_y() - B.getPosition_y())) * pz / FIELDY;
	double dr = sqrt(dx * dx + dy * dy);
	bool result = (dr < sqrt(A.getMass()) / pow(10, 11) / 5 + 0.5);

	return (result);
}
//given object A and object B that are colliding in a perfectly
//inelastic collision, finds the resulting object and velocity and
//stores the values in object A
MassObject* updateObjects(MassObject* A, MassObject* B) {
	double m1 = (*A).getMass();
	double m2 = (*B).getMass();
	double v1 = (*A).getvx();
	double v2 = (*B).getvx();
	double vx = (m1 * v1 + m2 * v2) / (m1 + m2);

	double v3 = (*A).getvy();
	double v4 = (*B).getvy();
	//cout << m1 << " " << m2 << endl;
	//cout << v3 << " " << v4 << endl;
	double vy = (m1 * v3 + m2 * v4) / (m1 + m2);
	//cout << vx << " " << vy << endl;
	(*A).changeV(vx, vy);
	(*A).changeMass((*B).getMass());

	return A;
}
//moves an object in the MassObject* A array to the back
//of the array, given the object's position in the array
void shiftObjects(MassObject* A, int position, int size) {
	MassObject temp = *(A + position);
	for (int k = position; k < size - 1; k++) {
		*(A + k) = *(A + k + 1);
	}
	*(A + size - 1) = temp;
}
//check to see if any collisions occured in an array of MassObjects
//If collision occurs, the larger of the two absorbs the smaller and conservation
//of momentum is applied.
//The smaller object of the array is moved to the back of the array and size is
//reduced by one.
//Assumption: array is sorted in order of largest mass to smallest mass
int check_collisions(MassObject* A, int size, int px, int pz) {
	for (int i = 0; i < size; i++) {
		for (int j = i + 1; j < size; j++) {
			MassObject* object1 = (A + i);
			MassObject* object2 = (A + j);
			if (isInRange(*object1, *object2, px, pz)) {
				//(m1*vx1+m2*vx2)/(m1+m2)
				cout << "Collision! ";
				cout << "Objects " << object1->getObjNumber() << " & " << object2->getObjNumber() << endl;
				object1 = updateObjects(object1, object2);
				//swap smaller object with last object, and shorten array
				shiftObjects(A, j, size);
				//cout << object1->getMass() << endl;
				size--;
			}
		}
	}
	return size;
}
//find the dx between two objects
double find_dx(MassObject A, MassObject B) {
	return (B.getPosition_x() - A.getPosition_x());
}
//find the dy between two objects
double find_dy(MassObject A, MassObject B) {
	return (B.getPosition_y() - A.getPosition_y());
}
//find the straight-line distance given dx and dy
double find_rsqrd(double dx, double dy) {
	double rsqrd = dx * dx + dy * dy;

	return rsqrd;
}
//find the acceleration by an object given that object and the distance
double set_acc(double rsqrd, MassObject A) {
	if (rsqrd == 0) {
		return 0;
	}

	double acc;
	acc = 6.67 * (A.getMass()) / rsqrd / 10000;

	return acc;
}
//finds the components of the acceleration given dx dy
void find_components(MassObject* A, double net_acc, double dx, double dy) {
	double theta = atan2(dy, dx);
	double ax = net_acc * cos(theta);
	double ay = net_acc * sin(theta);
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
				double dx = find_dx(*object1, *object2);
				double dy = find_dy(*object1, *object2);
				double rsqrd = find_rsqrd(dx, dy);
				double net_acc = set_acc(rsqrd, *object2);
				find_components(object1, net_acc, dx, dy);
			}
		}
	}
}

//generate a random decimal between fMin and fMax
double randDouble(double fMin, double fMax) {
	double f = ((double)rand() / (RAND_MAX));
	double i = (double)(rand() % (int)(fMax - fMin));
	double result = f + i;

	if (result < 0) {
		result *= -1;
	}
	return result;
}

//semi-randomly initialize the MassObjects given the field size and the number of objects
//all objects are randomly initialized with a mass between 10^22 kg to 10^24 kg
//40% of the objects will be initialized in a 2.5*10^10 by 2.5*10^10 field
//	centered in the middle of the frame
//The remaining 60% can spawn anywhere in the frame's field
void init(int px, int pz, int numberOfObjects, MassObject* arr) {
	int benchmark1 = numberOfObjects * 4 / 10;
	for (int i = 0; i < benchmark1; i++) {
		double x = (0.5 + randDouble(0, 2.5)) * pow(10, 10);
		double y = (0.5 + randDouble(0, 2.5)) * pow(10, 10);
		double vx = rand() % (500) - 250;
		vx *= pow(10, 3);
		double vy = rand() % (500) - 250;
		vy *= pow(10, 3);
		double mass = (rand() % 100 + 1) * pow(10, 22);
		*(arr + i) = MassObject(x, y, vx, vy, mass, i);
	}
	for (int i = benchmark1; i < numberOfObjects; i++) {
		double x = (randDouble(0, 4)) * pow(10, 10);
		double y = (randDouble(0, 4)) * pow(10, 10);
		double vx = rand() % (500) - 250;
		vx *= pow(10, 4);
		double vy = rand() % (500) - 250;
		vy *= pow(10, 4);
		double mass = (rand() % 100 + 1) * pow(10, 22);
		*(arr + i) = MassObject(x, y, vx, vy, mass, i);
	}
}
//test init used for debugging
void init2(int px, int pz, int numberOfObjects, MassObject* arr) {
	int benchmark1 = numberOfObjects * 4 / 10;
	int benchmark2 = numberOfObjects * 7 / 10;
	for (int i = 0; i < benchmark1; i++) {
		double x = (1.1 + randDouble(0, 1.8)) * pow(10, 10);
		double y = (1.1 + randDouble(0, 1.8)) * pow(10, 10);
		double vx = rand() % (500) - 250;
		vx *= pow(10, 3);
		double vy = rand() % (500) - 250;
		vy *= pow(10, 3);
		double mass = (rand() % 100 + 1) * pow(10, 22);
		*(arr + i) = MassObject(x, y, vx, vy, mass, i);
	}
	for (int i = benchmark1; i < benchmark2; i++) {
		double x = (0.5 + randDouble(0, 3)) * pow(10, 10);
		double y = (0.5 + randDouble(0, 3)) * pow(10, 10);
		double vx = rand() % (500) - 250;
		vx *= pow(10, 4);
		double vy = rand() % (500) - 250;
		vy *= pow(10, 4);
		double mass = (rand() % 100 + 1) * pow(10, 22);
		*(arr + i) = MassObject(x, y, vx, vy, mass, i);
	}
	for (int i = benchmark2; i < numberOfObjects; i++) {
		double x = (randDouble(0, 4)) * pow(10, 10);
		double y = (randDouble(0, 4)) * pow(10, 10);
		double vx = rand() % (500) - 250;
		vx *= pow(10, 4);
		double vy = rand() % (500) - 250;
		vy *= pow(10, 4);
		double mass = (rand() % 100 + 1) * pow(10, 22);
		*(arr + i) = MassObject(x, y, vx, vy, mass, i);
	}
}
bool validate_input(int px, int pz, int numberOfObjects) {
	if (px <= 0)
	{
		return false;
	}
	if (pz <= 0)
	{
		return false;
	}
	if (numberOfObjects <= 1)
	{
		return false;
	}
	return true;
}
int main() {

	srand(time(0));
	string arg_1 = "animation_project.bmp";
	int px;
	int pz;
	int numberOfObjects;
	double stepsize = 25;
	px = 800;
	pz = 800;
	numberOfObjects = 300;
	cout << "The frame width is " << px << "." << endl;
	cout << "The frame height is " << pz << "." << endl;
	cout << "The number of objects used is " << numberOfObjects << "." << endl;

	//test if input is valid; if invalid, program stops
	if (!validate_input) { return -1; }

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
	cout << "Buffer initialized." << endl;

	//initialize objects
	MassObject* arr = new MassObject[numberOfObjects];
	init(px, pz, numberOfObjects, arr);

	//define background color
	p_color b_color;
	b_color.r = 0;
	b_color.g = 0;
	b_color.b = 5;

	//draw frames

	//draw frame 0:
	fill_background(buffer, px, pz, b_color);
	for (int j = 0; j < numberOfObjects; j++) {
		struct r_circle thisObject;
		set_circle_values(thisObject, *(arr + j), px, pz);
		//cout << thisObject.cx << "  " << thisObject.cy << "  " << thisObject.radius << endl;
		fill_circle(buffer, px, pz, thisObject);
	}
	write_bmp_file(0, arg_1, buffer, px, pz);

	//draw the rest of the frames
	int size = numberOfObjects;
	for (int i = 1; i < NUMBEROFCYCLES; i++) {
		//sort the MassObject array
		sort_MassObjects(arr, 0, size);
		//check if any objects have collided
		size = check_collisions(arr, size, px, pz);

		//draw the objects
		fill_background(buffer, px, pz, b_color);
		for (int j = 0; j < size; j++) {
			struct r_circle thisObject;
			set_circle_values(thisObject, *(arr + j), px, pz);
			fill_circle(buffer, px, pz, thisObject);
		}
		//create a frame for every other cycle
		if (i % 2 == 0) {
			write_bmp_file(i / 2, arg_1, buffer, px, pz);
		}
		//check if any collisions have happened
		size = check_collisions(arr, size, px, pz);
		//update the accelerations of the objects
		calculateAccelerations(arr, size);
		//update each objects position
		for (int j = 0; j < size; j++) {
			(*(arr + j)).changePosition(stepsize);
		}
	}

	return 0;
}