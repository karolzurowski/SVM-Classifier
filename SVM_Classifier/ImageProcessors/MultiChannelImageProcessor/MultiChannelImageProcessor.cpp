

#include "MultiChannelImageProcessor.h"
#include <opencv2/contrib/contrib.hpp>

MultiChannelImageProcessor::MultiChannelImageProcessor(int meshGap, int meshWidth, int meshHeight) : ImageProcessorBase(
	meshGap, meshWidth, meshHeight)
{
}

Mat MultiChannelImageProcessor::ProcessImage(const Mat& image, const Mat& mask) const
{
	vector<Mat> channels(3);
//	Mat imageHSV;
//	cvtColor(image, imageHSV, CV_BGR2HSV);
	split(image, channels);

	vector<Mat> descriptors;
	for (auto channel : channels)
	{
		descriptors.push_back(CalculateSIFT(channel, mask));
	}

	Mat mergeChannels = Mat(descriptors[0]);
	for(int i=1;i<descriptors.size();i++)
		hconcat(mergeChannels, descriptors[i], mergeChannels);

	return mergeChannels;



}


