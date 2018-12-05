#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "../../Helpers/HelperStructs.h"

using namespace cv;
class ImageProcessorBase
{
public:
	ImageProcessorBase(int meshGap, int meshWidth, int meshHeight);

	virtual void CalculateSVMInput(const ImageGroup & images, SVMInput & svmInput)const;
	virtual void ProcessImage(const Mat& image, const Mat& mask, Mat& outputImage) const = 0;
	virtual void TestImage(const Mat& image, Mat& outputImage) const;
	Mat CalculateLabels(const Mat& maskDescriptors, const Mat& backgroundMaskDescriptors) const;
	vector<vector<KeyPoint>>  SplitAndCalculateKeyPoints(const Mat&image, const Mat& mask, int vectorLimit = 0) const;
	Mat Mesh() const { return mesh; }
	int RegionScale() const { return regionScale; }


protected:
	Mat mesh;
	SiftFeatureDetector siftDetector;
	int meshGap;

	void CalculateSIFT(const Mat& image, const Mat& mask,   Mat& outputImage) const;

	void CalculateKeyPoints(const Mat& image, vector<KeyPoint>& keyPoints) const;
	Mat CreateMesh(int meshGap, int meshWidth, int meshHeight) const;
	int imageWidth;
	int imageHeight;
	int regionScale = 20;



};

