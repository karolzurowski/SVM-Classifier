#pragma once
#include <opencv2/core/core.hpp>
#include <filesystem>
using namespace cv;
using namespace std;
using namespace filesystem;
struct ImageGroup
{
	Mat Image;
	Mat Mask;
	Mat BackgroundMask;
};

struct SVMInput
{
	Mat Data;
	Mat Labels;
};
struct ImagePath
{
	path DirectoryPath;
	path ImageFileName;
};