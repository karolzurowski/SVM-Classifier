#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "../../Helpers/HelperStructs.h"

using namespace cv;
class ImageProcessorBase
{
public:
	ImageProcessorBase(int imageWidth, int ImageHeight, int meshGap);	

	virtual void CalculateSVMInput(const ImageGroup & images, SVMInput & svmInput);
	virtual void ProcessImage(const Mat& image, const Mat& mask, Mat& outputImage)  = 0;
	virtual void ClassifyImage(const Mat& image, Mat& outputImage) ;
	Mat CalculateLabels(const Mat& maskDescriptors, const Mat& backgroundMaskDescriptors) const;
	
	virtual Mat Mesh() const { return mesh; }
	int RegionScale() const { return regionScale; }
	virtual void DrawResults(const vector<float>& results,  Mat& mat);


protected:
	Mat mesh;
	int meshGap;

	Mat CreateMesh(int meshGap, int meshWidth, int meshHeight) const;
	int imageWidth;
	int imageHeight;
	int regionScale =10;



};

