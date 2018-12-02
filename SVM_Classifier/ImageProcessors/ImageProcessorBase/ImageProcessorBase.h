#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "../../Helpers/HelperStructs.h"

using namespace cv;
class ImageProcessorBase
{
public:
	ImageProcessorBase(int meshGap, int meshWidth, int meshHeight);

	virtual SVMInput CalculateSVMInput(const ImageGroup & images)const;
	virtual Mat ProcessImage(const Mat& image, const Mat& mask) const = 0;
	virtual Mat ProcessImage(const Mat& image) const;	
	Mat CalculateLabels(const Mat& maskDescriptors, const Mat& backgroundMaskDescriptors) const;
	vector<vector<KeyPoint>>  SplitAndCalculateKeyPoints(const Mat&image, const Mat& mask, int vectorLimit = 0) const;
	Mat Mesh() const { return mesh; }
	int RegionScale() const { return regionScale; }

protected:
	Mat mesh;
	SiftFeatureDetector siftDetector;
	int meshGap;

	Mat CalculateSIFT(const Mat& image, const Mat& mask) const;

	vector<KeyPoint> CalculateKeyPoints(const Mat& image) const;
	Mat CreateMesh(int meshGap, int meshWidth, int meshHeight) const;
	int imageWidth;
	int imageHeight;
	int regionScale = 20;



};

