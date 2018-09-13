Sign Language Recognition using Neural Network Model
======================================================
Report Submitted by Cyril Naves - M1 International


Description:
=============
This Project is about integrating the Camshift code with Facedetect Code and them implement a neural network model to get the sign language recognition.

It is split into 4 parts:
1)CamShift integrated with FaceDetect
2)Modifying CamShift code to track the hand and extract its probabilistic mask from the video.
3)Training samples to a Multi-Layer Perceptron (MLP)
4)Recognition of the hand sign using the MLP Model.

Getting Started:
===================
Prerequisites:
===============
1) Run on the given PNS VM the ,SignDetection.cpp
2) Following are hardcoded values inside which has to be make sure it is present:
	string cascadeName = "/home/user/src/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml";
	string nestedCascadeName = "/home/user/src/opencv-3.3.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    string filename_to_load = "mlp_config.txt";
	
4) CascadeName and NestedCascadeName are used to load the classifiers from the haarcascades directory
5) mlp_config.txt is also present in this directory

Command to Compile:
====================
gcc -I /usr/local/include/ -L /usr/local/lib/ SignDetection.cpp -o SignDetection -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lstdc++ -lopencv_core -lopencv_videoio -lopencv_video -lopencv_ml

To Run the object file:
./SignDetection

Output:
==========
Displayed on the command terminal "DETECTED ALPHABET:"
In this model it is trained to recognize C and V alphabet.

This work was performed together with discussions on implementation with Ponathipan Jawahar but code refactoring and comments was performed individually.


