#pragma once
#include <filesystem>
#include <string>
#include <opencv2/imgproc.hpp>



using namespace std;
using namespace cv;


class MaskManager
{
public:
	
	MaskManager(const string& inputFilepath, const string& outputFilepath = "../Output_Masks");
	
	void SaveImage(const filesystem::path& imgPath, Mat sourceImg) const;
	Mat CalculateBackgroundMask(Mat & mat,int minMaskSize,int maxMaskSize) const;
	

	int InitialThreshold() const
	{
		return initialThreshold;
	}

	void InitialThreshold(int initialThreshold)
	{
		this->initialThreshold = initialThreshold;
	}
	
	void CreateBackgroundMasks();

private:
	vector<filesystem::path> imgPaths;
	filesystem::path outputDirectoryPath;
	int initialThreshold = 30;
	Mat CreateBackgroundMask(const filesystem::path& imgPath) const;
};
