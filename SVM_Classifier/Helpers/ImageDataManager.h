#pragma once
#include <filesystem>
#include <string>
#include <opencv2/imgproc.hpp>
#include "HelperStructs.h"


using namespace std;
using namespace cv;
using namespace filesystem;



class ImageDataManager
{
public:
	static Mat CalculateBackgroundMask(Mat& mat, int minMaskSize, int maxMaskSize);
	static vector<path> GetValidImageLists(path directoryPath, bool createBackgroundMasks = true);
	static void SaveImage(const path& imagePath, Mat sourceImage);
	static ImageGroup FetchImages(const path directoryPath, const path imageName,int flag = CV_LOAD_IMAGE_GRAYSCALE);

	/*static int InitialThreshold() { return initialThreshold; }
	static void InitialThreshold(int _initialThreshold) { initialThreshold = _initialThreshold; }*/

private:
	inline static int initialThreshold = 30;
	inline static const string masksDirectoryName = "\\Masks";
	inline static const string originalFramesDirectoryName = "\\Original_Frames";
	inline static const string backgroundMasksDirectoryName = "\\Background_Masks";

	static Mat CreateBackgroundMask(const filesystem::path& imgPath);
};
