#include "ImageProcessorBase.h"

ImageProcessorBase::ImageProcessorBase(int meshGap, int meshWidth, int meshHeight)
{
	this->meshGap = meshGap;
	mesh = CreateMesh(meshGap, meshWidth, meshHeight);
}

Mat ImageProcessorBase::ProcessImage(const Mat& image) const
{
	return	ProcessImage(image, mesh);
}

Mat ImageProcessorBase::CalculateSIFT(const Mat& image, const Mat& mask) const
{
	Mat descriptors;
	auto keyPoints = CalculateKeyPoints(mask);
	siftDetector.compute(image, keyPoints, descriptors);
	return descriptors;
}

Mat ImageProcessorBase::CalculateLabels(const Mat& maskDescriptors, const Mat& backgroundMaskDescriptors) const
{
	vector<float> labelsVector;
	for (int i = 0; i < maskDescriptors.rows; i++)
		labelsVector.push_back(1);
	for (int i = 0; i < backgroundMaskDescriptors.rows; i++)
		labelsVector.push_back(0);

	Mat labels(labelsVector);
	Mat labelsToReturn;
	labelsToReturn.push_back(labels);
	return labelsToReturn;
}

vector<KeyPoint> ImageProcessorBase::CalculateKeyPoints(const Mat& image) const
{
	vector<KeyPoint> keyPoints;
	vector<Point> imagePoints;
	Mat meshedImage;
	multiply(image, mesh, meshedImage);
	findNonZero(meshedImage, imagePoints);
	for (auto point : imagePoints)
	{
		keyPoints.push_back(KeyPoint(point, meshGap));
	}
	return keyPoints;
}

SVMInput ImageProcessorBase::CalculateSVMInput(const ImageGroup & images) const
{
	//todo assert images not null

	auto image = images.Image;
	auto mask = images.Mask;
	auto backgroundMask = images.BackgroundMask;

	Mat maskDescriptors = ProcessImage(image, mask);
	Mat backgroundMaskDescriptors = ProcessImage(image, backgroundMask);

	Mat descriptors;
	descriptors.push_back(maskDescriptors);
	descriptors.push_back(backgroundMaskDescriptors);

	Mat labelsToReturn = CalculateLabels(maskDescriptors, backgroundMaskDescriptors);

	return SVMInput{ descriptors,labelsToReturn };
}

Mat ImageProcessorBase::CreateMesh(int meshGap, int meshWidth, int meshHeight) const
{
	Mat mesh = Mat::zeros(meshHeight, meshWidth, CV_8UC1);

	for (size_t i = 0; i < meshHeight; i++)
	{
		if (i%meshGap == 0)
		{
			for (size_t j = 0; j < meshWidth; j++)
			{
				if (j%meshGap == 0)
					mesh.at<uchar>(i, j) = 255;
			}
		}
	}
	return mesh;
}


