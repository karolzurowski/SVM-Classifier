#pragma once
#include <opencv2/core/core.hpp>
#include "../SiftImageProcessor/SiftImageProcessor.h"

class BOWImageProcessor :public SiftImageProcessor
{
public:
	BOWImageProcessor( int imageWidth, int imageHeight, int meshGap);
	BOWImageProcessor(Mat &dictionary,  int imageWidth, int imageHeight, int meshGap);
	void ProcessImage(const Mat& image, const Mat& mask, Mat& outputImage) override;
	Mat CreateBowDictionary(const vector<ImagePath>& trainPaths,int dictionarySize);
	Mat BowDictionary() const { return  hasBowDictionary ? bowDictionary : Mat(); }


	Mat Mesh() const override { return Mat(); }
	void DrawResults(const vector<float>& results, Mat& mat) override;
private:
	Mat bowDictionary;
	bool hasBowDictionary;
	void BowDictionary(const Mat& bowDictionary)
	{
		this->bowDictionary = bowDictionary;
		hasBowDictionary = true;
	}
	void SaveDictionary();
	Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher;
	Ptr<FeatureDetector> detector = new SiftFeatureDetector;
	Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor;
	BOWImgDescriptorExtractor bowImgDescriptorExtractor;
	vector<vector<KeyPoint>>  SplitAndCalculateKeyPoints(const Mat&image, const Mat& mask, int vectorLimit = 0) const;
	//int patchScale = 20;
};

