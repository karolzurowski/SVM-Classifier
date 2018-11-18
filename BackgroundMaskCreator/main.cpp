#include <string>
#include <iostream>
#include <filesystem>
#include <opencv2/highgui.hpp>

#include "mask_manager.h"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
	string inputPath = "../Test_Masks";
  //  string outputPathString = "C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\metal_cylinder_2\\Background_Masks";	

	try
	{
		MaskManager maskManager(inputPath);
		maskManager.CreateBackgroundMasks();
	}
	catch (filesystem::filesystem_error& error)
	{
		cout << error.what();
	}
	
	waitKey(0);
	return 0;
}
	