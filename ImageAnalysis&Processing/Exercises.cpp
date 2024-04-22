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

//void Exercise2(const cv::Mat& image)
//{
//	if (image.empty()) {
//		std::cerr << "Error: Couldn't load the binary image." << std::endl;
//		exit(EXIT_FAILURE);
//	}
//
//	cv::threshold(image, image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
//
//	std::vector<std::vector<cv::Point>> contours;
//	cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//	for (size_t i = 0; i < contours.size(); ++i) {
//		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
//		cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
//
//		double area = computeArea(mask);
//		int circumference = computeCircumference(mask);
//
//		cv::drawContours(image, contours, static_cast<int>(i), cv::Scalar(128), 2);
//
//		cv::Rect bbox = cv::boundingRect(contours[i]);
//
//		cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
//
//		cv::putText(image, "Img" + std::to_string(i + 1), center, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, cv::LINE_AA);
//		cv::putText(image, "Area: " + std::to_string(area), cv::Point(center.x, center.y + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, cv::LINE_AA);
//		cv::putText(image, "Circumference: " + std::to_string(circumference), cv::Point(center.x, center.y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1, cv::LINE_AA);
//
//		computeFeatures(mask, i + 1);
//	}
//
//	cv::Mat widenedImage;
//	cv::copyMakeBorder(image, widenedImage, 0, 0, 0, 200, cv::BORDER_CONSTANT, cv::Scalar(0));
//
//	cv::imshow("Result", widenedImage);
//	cv::waitKey(0);
//}

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
	classifier.computeEthalons(trainingImages, labels);

	// Classify each object in the image
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); ++i) {
		// Convert contour to a binary mask
		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

		// Classify the shape using the EtalonClassifier
		std::string shape = classifier.classifyShape(mask);

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
	std::vector<cv::Mat> trainingImages;
	std::vector<std::string> labels;

	// Compute features for each object in the image
	std::vector<cv::Vec2d> objectFeatures;

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); ++i) {
		// Convert contour to a binary mask
		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

		// Compute features for the object
		cv::Vec2d objectFeature = classifier.getFeatures(mask);
		objectFeatures.push_back(objectFeature);

		// Assign a unique label to each object
		labels.push_back("object_" + std::to_string(i));
	}

	// Compute ethalons for each class
	classifier.computeEthalons(objectFeatures, labels);

	// Save ethalons
	classifier.saveEthalons("ethalons.dat");

	// Classify objects in the test image
	for (size_t i = 0; i < contours.size(); ++i) {
		// Convert contour to a binary mask
		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

		// Classify the shape using the EtalonClassifier
		std::string shape = classifier.classifyShape(mask);

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

//void Exercise3(const cv::Mat& image)
//{
//	// Check if the input image is empty
//	if (image.empty()) {
//		std::cerr << "Error: Input image is empty." << std::endl;
//		exit(EXIT_FAILURE);
//	}
//
//	// Print the type of the input image
//	std::cout << "Input image type: " << image.type() << std::endl;
//
//	// Display the input image
//	cv::imshow("Input Image", image);
//	cv::waitKey(0);
//
//	// Check if the input image is of type CV_8UC1 (single-channel 8-bit)
//	if (image.type() != CV_8UC1) {
//		// Convert the input image to grayscale
//		cv::Mat grayscaleImage;
//		cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);
//
//		// Print the type of the grayscale image
//		std::cout << "Grayscale image type: " << grayscaleImage.type() << std::endl;
//
//		// Display the grayscale image
//		cv::imshow("Grayscale Image", grayscaleImage);
//		cv::waitKey(0);
//
//		// Continue with the grayscale image
//		processImage(grayscaleImage);
//	}
//	else {
//		// Continue with the original input image
//		processImage(image);
//	}
//}

//void processImage(const cv::Mat& image)
//{
//	// In Exercise 3, we will use the EtalonClassifier class to classify objects in the image.
//
//	// Load the test image
//	cv::Mat testImage = image;
//
//	// Load the training images
//	std::vector<cv::Mat> trainingImages;
//	std::vector<std::string> labels;  // Add this line
//
//	trainingImages.push_back(cv::imread("C:\\Users\\Lenovo\\Downloads\\etalon.png", cv::IMREAD_GRAYSCALE));
//	labels.push_back("square"); 
//	labels.push_back("rectangle");
//	labels.push_back("star");
//
//	// Create an instance of the EtalonClassifier class
//	EtalonClassifier classifier;
//
//	// Compute the ethalons
//	classifier.computeEthalons(trainingImages, labels);  // Provide both trainingImages and labels
//
//	// Classify the object in the test image
//	std::string result = classifier.classifyObject(testImage);
//
//	// Print the result
//	std::cout << "Classified object as: " << result << std::endl;
//
//	// Display the test image
//	cv::imshow("Test Image", testImage);
//	cv::waitKey(0);
//}

void detectAndColorizeObjects(cv::Mat& image, EtalonClassifier& classifier)
{
	// Convert the image to grayscale
	cv::Mat grayscaleImage;
	cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);

	// Threshold the grayscale image
	cv::Mat binaryImage;
	cv::threshold(grayscaleImage, binaryImage, 128, 255, cv::THRESH_BINARY);

	// Classify objects in the binary image
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Iterate through each contour and classify the shape
	for (const auto& contour : contours) {
		// Convert contour to a binary mask
		cv::Mat mask = cv::Mat::zeros(binaryImage.size(), CV_8UC1);
		cv::drawContours(mask, std::vector<std::vector<cv::Point>>(1, contour), -1, cv::Scalar(255), cv::FILLED);

		// Compute the bounding box of the contour
		cv::Rect bbox = cv::boundingRect(contour);

		// Compute the aspect ratio of the bounding box
		double aspectRatio = (double)bbox.width / bbox.height;

		// Skip the contour if the aspect ratio is too large or too small
		if (aspectRatio > 2.0 || aspectRatio < 0.5) {
			continue;
		}

		// Compute the features of the contour
		cv::Vec2d features = classifier.getFeatures(mask);


		// Compute the distance to the ethalons
		double minDistance = std::numeric_limits<double>::max();

		// Classify the shape
		std::string shape = classifier.classifyShape(mask);

		// Colorize the contour based on its shape
		cv::Scalar color;
		if (shape == "square") {
			color = cv::Scalar(255, 0, 0); // Blue
		}
		else if (shape == "rectangle") {
			color = cv::Scalar(0, 255, 0); // Green
		}
		else if (shape == "star") {
			color = cv::Scalar(0, 0, 255); // Red
		}
		else {
			// Unknown shape
			continue;
		}

		// Draw the contour with the identified shape color
		cv::drawContours(image, std::vector<std::vector<cv::Point>>(1, contour), -1, color, 2);
	}

	// Display the result
	cv::imshow("Result", image);
	cv::waitKey(0);
}



void processImage(const cv::Mat& image)
{
	// Create an instance of the EtalonClassifier class
	EtalonClassifier classifier;

	// Load the training images and labels
	std::vector<cv::Mat> trainingImages;
	trainingImages.push_back(cv::imread("C:\\Users\\Lenovo\\Downloads\\square.png", cv::IMREAD_GRAYSCALE));
	trainingImages.push_back(cv::imread("C:\\Users\\Lenovo\\Downloads\\rectangle.jpg", cv::IMREAD_GRAYSCALE));
	trainingImages.push_back(cv::imread("C:\\Users\\Lenovo\\Downloads\\star-red.jpg", cv::IMREAD_GRAYSCALE));

	std::vector<std::string> labels = { "square", "rectangle", "star" };

	// Compute ethalons
	classifier.computeEthalons(trainingImages, labels);

	// Save ethalons to a file
	classifier.saveEthalons("ethalons.yml");

	// Load ethalons from a file
	classifier.loadEthalons("ethalons.yml");

	// Classify the shape of the test 

	// Detect and colorize objects in the test image
	cv::Mat testImage = image.clone();
	detectAndColorizeObjects(testImage, classifier);

	// Classify objects using ethalons
	std::string result = classifier.classifyObject(testImage);
	std::cout << "Classified object: " << result << std::endl;

	// Display the result
	//cv::imshow("Result", testImage);
	cv::waitKey(0);
}


