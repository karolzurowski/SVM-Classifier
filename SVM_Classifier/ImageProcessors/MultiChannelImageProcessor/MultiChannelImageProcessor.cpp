

#include "MultiChannelImageProcessor.h"
#include <opencv2/contrib/contrib.hpp>

MultiChannelImageProcessor::MultiChannelImageProcessor(int meshGap, int meshWidth, int meshHeight) : ImageProcessorBase(
	meshGap, meshWidth, meshHeight)
{
}

void MultiChannelImageProcessor::ProcessImage(const Mat& image, const Mat& mask,Mat& outputImage) const
{
	vector<Mat> channels(3);
	Mat imageHSV;
//	cvtColor(image, imageHSV, CV_BGR2HSV);
	split(image, channels);

	vector<Mat> descriptors;
	for (auto channel : channels)
	{
		Mat channelDescriptors;
		CalculateSIFT(channel, mask, channelDescriptors);

		descriptors.push_back(channelDescriptors);
	}

	outputImage = Mat(descriptors[0]);
	for(int i=1;i<descriptors.size();i++)
		hconcat(outputImage, descriptors[i], outputImage);



}


