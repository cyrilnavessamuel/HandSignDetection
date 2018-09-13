#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv/cv.hpp>
#include <fstream>
#include "opencv2/ml.hpp"


using namespace std;
using namespace cv;
using namespace cv::ml;

static void help()
{
	cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
		"This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
		"It's most known use is for faces.\n"
		"Usage:\n"
		"./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
		"   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
		"   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
		"   [--try-flip]\n"
		"   [filename|camera_index]\n\n"
		"see facedetect.cmd for one call:\n"
		"./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
		"During execution:\n\tHit any key to quit.\n"
		"\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, vector<Rect>&faces);
Mat myCamshift(Rect& selection, Mat& image);




/*
This method loads classifier which contains the trained MLP parameters from the specified file argumennts
*/
template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
	//Loads the complete MLP model state from the file
	Ptr<T> model = StatModel::load<T>(filename_to_load);
	//Check if model is loaded
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;
	else
		cout << "The classifier " << filename_to_load << " is loaded.\n";

	return model;
}
/*
Main function to run the Face Detection Program with command line arguments to pass cascadeName,nestedCascadeName,scale,try-flip,input arguments
*/
int main(int argc, const char** argv)
{
	//VideoCapture Class for playing video wher face needs to be detected
	VideoCapture capture;
	Mat frame, image;
	string inputName;
	bool tryflip;
	//Predefined Trained XML Classifers with Facial features
	CascadeClassifier cascade, nestedCascade;
	double scale;
	// Classifers from haarcascade directory for frontal face
	string cascadeName = "/home/user/src/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml";
	// Classifier from haarcascade directory for frontal eye detector with better handling of eyeglassess
	string nestedCascadeName = "/home/user/src/opencv-3.3.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
	cv::CommandLineParser parser(argc, argv,
		"{help h||}"
		"{cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
		"{nested-cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
		"{scale|1|}{try-flip||}{@filename||}"
	);
	if (parser.has("help"))
	{
		help();
		return 0;
	}

	scale = parser.get<double>("scale");
	if (scale < 1)
		scale = 1;
	tryflip = parser.has("try-flip");
	inputName = parser.get<string>("@filename");
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	//Loads the cascade files of nestedCascade and Cascade
	if (!nestedCascade.load(nestedCascadeName))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		help();
		return -1;
	}
	//Gets the input video streaming from camera
	if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
	{
		int camera = inputName.empty() ? 0 : inputName[0] - '0';
		if (!capture.open(camera))
			cout << "Capture from camera #" << camera << " didn't work" << endl;
	}
	else if (inputName.size())
	{
		image = imread(inputName, 1);
		if (image.empty())
		{
			if (!capture.open(inputName))
				cout << "Could not read " << inputName << endl;
		}
	}
	//Load the static image
	else
	{
		image = imread("../data/lena.jpg", 1);
		if (image.empty()) cout << "Couldn't read ../data/lena.jpg" << endl;
	}
	/*Declare variable faces */
	vector<Rect> faces;

	//Capture the video
	if (capture.isOpened())
	{
		cout << "Video capturing has been started ..." << endl;
		//Declaration of our Artificial Neural Network MultiLayer Perceptron " Ptr -> Template class for smart pointers which hold pointers of ANN_MLP 
		Ptr<ANN_MLP> model;
		//MLP File Configuration Input File
		string filename_to_load = "mlp_config.txt";
		//load classifer from our trained MLP config file
		model = load_classifier<ANN_MLP>(filename_to_load);
		//Infinite loop to continuously capture the video frame
		for (;;)
		{
			capture >> frame;
			//Stop the infinite loop when no more input
			if (frame.empty())
				break;

			//Perform a deep copy of the input matrices of the frame
			Mat frame1 = frame.clone();

			//Check if faces Rect Vector is empty and then populate "faces - Rect Vector" by calling detectandDraw() function
			if (faces.empty()) {
				detectAndDraw(frame1, cascade, nestedCascade, scale, tryflip, faces);

			}
			//Else pass it to my myCamshift() method to get the matrices of the hand image to be predicted
			else {
				Mat img = myCamshift(faces[0], frame1);
				faces.clear();
				//convert image values to 4 byte floating points of to covert the pixel values
				img.convertTo(img, CV_32F);
				//Get a prediction from the model by passing our  hand symbol image matrix
				float r = model->predict(img);
				//Add the prediction to the alphabet A to get desired alphabets
				char out = r + 'A';
				cout << "R : " << r << endl;
				cout << "DETECTED ALPHABET : " << out << endl;
			}
			char c = (char)waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}
	else
	{
		cout << "Video not capturing" << endl;
	}

	return 0;
}
/*
This function is used to detect the face and populate the faces Vector of Rect
Input Arguments:
img ->  reference variable to our input frame
cascade ->Loaded Cascade classifier reference
nestedCascade -> Loaded Nested Cascade classifier
scale -> scale of the image
faces -> reference variable of the vector of Rect to be populated
*/
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, vector<Rect>&faces)
{
	double t = 0;
	vector<Rect>  faces2;
	//Scalar array for getting the color of the circles to be drawn around the image
	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};
	Mat gray, smallImg;
	//Convert the color of the input image to gray
	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	//resize the image to the input scale
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	//Equalizes the histogram of a grayscale image
	equalizeHist(smallImg, smallImg);
	//Gets the no of ticks to measure a function execution time
	t = (double)getTickCount();
	//Detect objects of different sizes in smallImg by passing the values of the detected objects as rectangles to the faces vector
	//1.1 -> scalefactor(How much image is reduced at each image scale) ; 2 ->minNeighbors(how much neighbors a rectangle should have to retain it) ; 
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		| CASCADE_SCALE_IMAGE,
		Size(30, 30));
	//Executed only flip argument is set to true to and flip the image along y axis
	if (tryflip)
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2,
			1.1, 2, 0
			| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)getTickCount() - t;
	//Detection time for the image
	printf("detection time = %g ms\n", t * 1000 / getTickFrequency());
	//Iterate through detected Rect of faces and draw the circles and the rectangle around the image
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double)r.width / r.height;
		//This aspect ratio is chosen to detect the nose between 0.75 and 1.3
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
				cvPoint(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
				color, 3, 8, 0);
		if (nestedCascade.empty())
			continue;
		//Select a region of Interest of the rect
		smallImgROI = smallImg(r);
		//Pass the Smaller Region of interest to nested cascade to detect the eyes with spectacles
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 2, 0
			| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		//For each nested Object which is the eye and the spectacles draw the circle.
		for (size_t j = 0; j < nestedObjects.size(); j++)
		{
			Rect nr = nestedObjects[j];
			center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
			center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
			radius = cvRound((nr.width + nr.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
	}
}

/*
This function performs the masking of the face in the backproj
*/
Mat myCamshift(Rect& selection, Mat& image)
{
	int trackObject = -1;
	int vmin = 10, vmax = 256, smin = 30;
	RotatedRect trackBox;
	Rect trackWindow;
	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;
	Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;

	//converting image to hsv image color space
	cvtColor(image, hsv, COLOR_BGR2HSV);
	if (trackObject) {
		int _vmin = vmin, _vmax = vmax;

		inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
			Scalar(180, 256, MAX(_vmin, _vmax)), mask);
		int ch[] = { 0, 0 };
		hue.create(hsv.size(), hsv.depth());
		//Copies channel of hue into hsv
		mixChannels(&hsv, 1, &hue, 1, ch, 1);
		if (trackObject < 0) {

			// Get the region of interest
			Mat roi(hue, selection), maskroi(mask, selection);
			//Calculate the histogram range 
			calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize,
				&phranges);
			normalize(hist, hist, 0, 255, NORM_MINMAX);

			trackWindow = selection;
			// int variable to avoid the Region of interested being selected again
			trackObject = 1; 

			histimg = Scalar::all(0);
			int binW = histimg.cols / hsize;
			Mat buf(1, hsize, CV_8UC3);
			for (int i = 0; i < hsize; i++)
			buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255, 255);
			cvtColor(buf, buf, COLOR_HSV2BGR);

			for (int i = 0; i < hsize; i++) {
				int val = saturate_cast<int>(hist.at<float>(i) * histimg.rows / 255);
				rectangle(histimg, Point(i * binW, histimg.rows),
					Point((i + 1) * binW, histimg.rows - val),
					Scalar(buf.at<Vec3b>(i)), -1, 8);
			}
		}

		// Calculate the backproject of the histogram
		calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
		backproj &= mask;
		//Blocking the face by putting the backproj to 0
		int face_to_end = image.cols - selection.x;
		trackWindow.x = 1;
		trackWindow.y = 1;
		trackWindow.height = backproj.cols - 1;
		trackWindow.width = backproj.rows - 1;
		//Fill to 0 in order to hide the face along the Matrices of backproj

		for (int i = 0; i < face_to_end; i++) {
			for (int j = 0; j < image.cols; j++) {
				backproj.at<char>(0 + j, selection.x + i) = 0;

			}
		}
		for (int i = image.rows - 100; i < image.rows; i++) {

			for (int j = 0; j < backproj.cols; j++) {
				backproj.at<char>(i, j) = 0;
			}
		}
		//This implements the camshift object tracking algorithm, finding an object center,optimal rotation returning a Rotated Rect
		RotatedRect trackBox = CamShift(backproj, trackWindow,
			TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
		if (trackWindow.area() <= 1)
		{
			int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
			trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
				trackWindow.x + r, trackWindow.y + r) &
				Rect(0, 0, cols, rows);
		}
		if (waitKey(10) == 32) {

			imshow("CamShift Demo", image);
			//Convert the Rotated Rect back to Rect inorder to get the new back proj
			Rect newTrackBox = trackBox.boundingRect();
			newTrackBox &= Rect(0, 0, backproj.cols, backproj.rows);

			Mat newHand = backproj(newTrackBox);
			resize(newHand, newHand, Size(16, 16), 0, 0, INTER_LINEAR);
			imshow("Hand", newHand);
			Mat img = newHand.reshape(0, 1);
			ofstream os("letter-train.txt", ios::out | ios::app);
			os << "V,";
			os << format(img, Formatter::FMT_CSV) << endl;
			os.close();
			cout << "Store the alphabet in text file" << endl;

		}
		cvtColor(backproj, image, COLOR_GRAY2BGR);
		ellipse(image, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
	}



	imshow("CamShift Demo", image);
	Rect newTrackBox = trackBox.boundingRect();
	newTrackBox &= Rect(0, 0, backproj.cols, backproj.rows);

	Mat hand2 = backproj(newTrackBox);
	resize(hand2, hand2, Size(16, 16), 0, 0, INTER_LINEAR);
	imshow("new hand", hand2);
	Mat img = hand2.reshape(0, 1);
	return img;
}
