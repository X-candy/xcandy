#include <Windows.h>
#include <process.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
#if _DEBUG
#pragma comment(lib,"opencv_core247d.lib")
#pragma comment(lib,"opencv_imgproc247d.lib")
#pragma comment(lib,"opencv_highgui247d.lib")
#pragma comment(lib,"opencv_ml247d.lib")
#pragma comment(lib,"opencv_photo247d.lib")
#pragma comment(lib,"opencv_video247d.lib")
#pragma comment(lib,"opencv_features2d247d.lib")
#pragma comment(lib,"opencv_calib3d247d.lib")
#pragma comment(lib,"opencv_objdetect247d.lib")
#pragma comment(lib,"opencv_contrib247d.lib")
#pragma comment(lib,"opencv_legacy247d.lib")
#pragma comment(lib,"opencv_flann247d.lib")
#else
#pragma comment(lib,"opencv_core247.lib")
#pragma comment(lib,"opencv_imgproc247.lib")
#pragma comment(lib,"opencv_highgui247.lib")
#pragma comment(lib,"opencv_ml247.lib")
#pragma comment(lib,"opencv_video247.lib")
#pragma comment(lib,"opencv_photo247.lib")
#pragma comment(lib,"opencv_features2d247.lib")
#pragma comment(lib,"opencv_calib3d247.lib")
#pragma comment(lib,"opencv_objdetect247.lib")
#pragma comment(lib,"opencv_contrib247.lib")
#pragma comment(lib,"opencv_legacy247.lib")
#pragma comment(lib,"opencv_flann247.lib")
#endif

