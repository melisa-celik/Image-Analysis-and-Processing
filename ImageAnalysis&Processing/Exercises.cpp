#include "Exercises.h"
#include "Functions.h"

void Exercise1(const cv::Mat& image)
{
	// load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::imshow("Lena", image);

	// declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits(uchar)
	cv::Mat gray_8uc1_img;
	// declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits(float)
	cv::Mat gray_32fc1_img;

	cv::cvtColor(image, gray_8uc1_img, cv::COLOR_BGR2GRAY);
	// convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	gray_8uc1_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0 / 255.0);
	// convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0

	int x = 10, y = 15;
	// pixel coordinates

	uchar p1 = gray_8uc1_img.at<uchar>(y, x);
	// read grayscale value of a pixel, image represented using 8 bits
	float p2 = gray_32fc1_img.at<float>(y, x);
	// read grayscale value of a pixel, image represented using 32 bits
	cv::Vec3b p3 = image.at<cv::Vec3b>(y, x);
	// read color value of a pixel, image represented using 8 bits per color channel

	// print values of pixels
	printf("p1 = %d\n", p1);
	printf("p2 = %f\n", p2);
	printf("p3[ 0 ] = %d, p3[ 1 ] = %d, p3[ 2 ] = %d\n", p3[0], p3[1], p3[2]);

	gray_8uc1_img.at<uchar>(y, x) = 0;
	// set pixel value to 0 (black)

	// draw a rectangle
	cv::rectangle(gray_8uc1_img, cv::Point(65, 84), cv::Point(75, 94), cv::Scalar(50), cv::FILLED);

	// declare variable to hold gradient image with dimensions: width= 256 pixels, height = 50 pixels.
	// Gray levels wil be represented using 8 bits (uchar)
	cv::Mat gradient_8uc1_img(50, 256, CV_8UC1);

	// For every pixel in image, assign a brightness value according to the x coordinate.
	// This wil create a horizontal gradient.
	for (int y = 0; y < gradient_8uc1_img.rows; y++)
	{
		for (int x = 0; x < gradient_8uc1_img.cols; x++)
		{
			gradient_8uc1_img.at<uchar>(y, x) = x;
		}
	}

	// diplay images
	cv::imshow("Gradient 8uc1", gradient_8uc1_img);
	cv::imshow("Lena gray 8uc1", gray_8uc1_img);
	cv::imshow("Lena gray 32fc1", gray_32fc1_img);

	cv::waitKey(0);
	// wait until keypressed
}

void Exercise2(const cv::Mat& image)
{
	// Check if the input image is empty
	if (image.empty()) {
		std::cerr << "Error: Couldn't load the binary image." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Convert the image to grayscale
	cv::Mat grayscaleImage;
	cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);

	// Threshold the grayscale image
	cv::Mat binaryImage;
	cv::threshold(grayscaleImage, binaryImage, 128, 255, cv::THRESH_BINARY);

	// Initialize the EtalonClassifier
	EtalonClassifier classifier;

	// Prepare training images and corresponding labels
	std::vector<cv::Mat> trainingImages = {
		/* Add your training images here */
	};
	std::vector<std::string> labels = {
		/* Add corresponding labels for each training image */
	};

	// Compute ethalons for each class
	//classifier.computeEthalons(trainingImages, labels);

	// Classify each object in the image
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); ++i) {
		// Convert contour to a binary mask
		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

		// Classify the shape using the EtalonClassifier
		std::string shape = classifier.classifyObject(mask);

		// Draw contour on the original image
		cv::drawContours(image, contours, static_cast<int>(i), cv::Scalar(128), 2);

		// Compute area and circumference of the contour
		double area = computeArea(mask);
		int circumference = computeCircumference(mask);

		// Compute bounding box and center
		cv::Rect bbox = cv::boundingRect(contours[i]);
		cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);

		// Add shape label to the image
		cv::putText(image, shape, center, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, cv::LINE_AA);
		cv::putText(image, "Area: " + std::to_string(area), cv::Point(center.x, center.y + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, cv::LINE_AA);
		cv::putText(image, "Circumference: " + std::to_string(circumference), cv::Point(center.x, center.y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, cv::LINE_AA);

		// Colorize the object based on its shape classification
		if (shape == "square") {
			cv::drawContours(image, contours, static_cast<int>(i), cv::Scalar(255, 0, 0), cv::FILLED);
		}
		else if (shape == "rectangle") {
			cv::drawContours(image, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), cv::FILLED);
		}
		else if (shape == "star") {
			cv::drawContours(image, contours, static_cast<int>(i), cv::Scalar(0, 0, 255), cv::FILLED);
		}
	}

	// Display the result
	cv::Mat widenedImage;
	cv::copyMakeBorder(image, widenedImage, 0, 0, 0, 200, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::imshow("Result", widenedImage);
	cv::waitKey(0);
}

void Exercise3(const cv::Mat& image)
{
	// Check if the input image is empty
	if (image.empty()) {
		std::cerr << "Error: Couldn't load the test image." << std::endl;
		return;
	}

	// Convert the test image to grayscale
	cv::Mat grayscaleImage;
	cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);

	// Threshold the grayscale image to obtain a binary image
	cv::Mat binaryImage;
	cv::threshold(grayscaleImage, binaryImage, 128, 255, cv::THRESH_BINARY);

	EtalonClassifier classifier;

	// Prepare training images and corresponding labels for the Etalon classifier
	std::vector<cv::Mat> trainingImages = {
		cv::imread("C:\\Users\\Lenovo\\Downloads\\square.png", cv::IMREAD_GRAYSCALE),
		cv::imread("C:\\Users\\Lenovo\\Downloads\\rectangle.jpg", cv::IMREAD_GRAYSCALE),
		cv::imread("C:\\Users\\Lenovo\\Downloads\\star-red.jpg", cv::IMREAD_GRAYSCALE)
	};
	std::vector<std::string> labels = { "square", "rectangle", "star" };

	classifier.computeEthalons(trainingImages, labels);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); ++i) {
		std::vector<cv::Point> currentContour = contours[i];

		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

		std::string shape = classifier.classifyObject(mask);

		std::cout << "Shape classification: " << shape << std::endl;

		double area = cv::contourArea(currentContour);
		double circumference = cv::arcLength(currentContour, true);

		cv::drawContours(image, contours, static_cast<int>(i), cv::Scalar(128), 2);

		cv::Rect bbox = cv::boundingRect(currentContour);
		cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);

		std::string objectText = "Object " + std::to_string(i + 1) + ": " + shape;
		cv::putText(image, objectText, cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1, cv::LINE_AA);
		//cv::putText(image, "Area: " + std::to_string(area), cv::Point(bbox.x, bbox.y + bbox.height + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, cv::LINE_AA);
		//cv::putText(image, "Circumference: " + std::to_string(circumference), cv::Point(bbox.x, bbox.y + bbox.height + 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, cv::LINE_AA);

		if (shape == "square") {
			cv::drawContours(image, contours, static_cast<int>(i), cv::Scalar(255, 0, 0), cv::FILLED);
		}
		else if (shape == "rectangle") {
			cv::drawContours(image, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), cv::FILLED);
		}
		else if (shape == "star") {
			cv::drawContours(image, contours, static_cast<int>(i), cv::Scalar(0, 0, 255), cv::FILLED);
		}

		cv::imshow("Shape Classification Result", image);
		cv::waitKey(1000); 

		mask.setTo(0);
	}

	cv::imshow("Final Result", image);
	cv::waitKey(0);
}
