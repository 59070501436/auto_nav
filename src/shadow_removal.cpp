#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/Marker.h>

#include <ros/ros.h>
#include <iostream>
#include <sstream>
#include <cmath>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "shadow_removal");  // initialize ROS and the node
    ros::NodeHandle nh;

    //Input and Output Image;
    Mat gscale, gscale_rgb, rgb_img, hsv_img;

    //Load the image
    string homedir = getenv("HOME");
        //cout << homedir << endl;
    rgb_img = cv::imread("/home/saga/frogn_fields_007.jpg"); // RGB Image

    Mat channel[3];

    // The actual splitting.
    split(rgb_img, channel);
    gscale = 3*channel[1]-channel[0]-channel[2];
    //cvtColor(gscale, gscale_rgb, CV_GRAY2RGB);

    threshold(gscale, gscale, 127, 255, cv::THRESH_OTSU);
    threshold(gscale, gscale, 127, 255, cv::THRESH_BINARY_INV+cv::THRESH_OTSU);

    Mat eroded, dil;
    Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  // needed for morphological transforms (erodation, dilation)
    erode(gscale, eroded, element); // eroding + dilating = opening
    dilate(eroded, gscale, element);
    // Canny(wscale, wscale, 50, 200);

    medianBlur(gscale,gscale,5);
    
    imwrite("/home/saga/green_channel.png", gscale);


}
