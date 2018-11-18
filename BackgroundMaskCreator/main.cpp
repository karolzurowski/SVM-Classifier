#include <string>
#include <iostream>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv/ml.h>
#include "mask_manager.h"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
	string inputPath = "C:\\Users\\karol\\Desktop\\toilet";
    string outputPathString = "C:\\Users\\karol\\Desktop\\toilet\\output";	

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
	