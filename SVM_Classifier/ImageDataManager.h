#pragma once
#include <filesystem>
#include <string>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;


struct ImageGroup
{
	Mat OriginalImage;
	Mat Mask;
	Mat BackgroundMask;
};


class ImageDataManager
{
public:
	static Mat CalculateBackgroundMask(Mat& mat, int minMaskSize, int maxMaskSize);
	static vector<filesystem::path> GetValidImageLists(string directoryPath, bool createBackgroundMasks = true);
	static void SaveImage(const filesystem::path& imagePath, Mat sourceImage);
	static ImageGroup FetchImages(const filesystem::path directoryPath, string imageName);

	/*static int InitialThreshold() { return initialThreshold; }
	static void InitialThreshold(int _initialThreshold) { initialThreshold = _initialThreshold; }*/

private:
	inline static int initialThreshold = 30;
	inline static const string masksDirectoryName = "\\Masks";
	inline static const string originalFramesDirectoryName = "\\Original_Frames";
	inline static const string backgroundMasksDirectoryName = "\\Background_Masks";

	static Mat CreateBackgroundMask(const filesystem::path& imgPath);
};
