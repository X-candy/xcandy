
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>

using namespace cv;
using namespace std;

#if _DEBUG
#pragma comment(lib,"opencv_core245d.lib")
#pragma comment(lib,"opencv_imgproc245d.lib")
#pragma comment(lib,"opencv_highgui245d.lib")
#pragma comment(lib,"opencv_ml245d.lib")
#pragma comment(lib,"opencv_photo245d.lib")
#pragma comment(lib,"opencv_video245d.lib")
#pragma comment(lib,"opencv_features2d245d.lib")
#pragma comment(lib,"opencv_calib3d245d.lib")
#pragma comment(lib,"opencv_objdetect245d.lib")
#pragma comment(lib,"opencv_contrib245d.lib")
#pragma comment(lib,"opencv_legacy245d.lib")
#pragma comment(lib,"opencv_flann245d.lib")
#else
#pragma comment(lib,"opencv_core245.lib")
#pragma comment(lib,"opencv_imgproc245.lib")
#pragma comment(lib,"opencv_highgui245.lib")
#pragma comment(lib,"opencv_ml245.lib")
#pragma comment(lib,"opencv_video245.lib")
#pragma comment(lib,"opencv_photo245.lib")
#pragma comment(lib,"opencv_features2d245.lib")
#pragma comment(lib,"opencv_calib3d245.lib")
#pragma comment(lib,"opencv_objdetect245.lib")
#pragma comment(lib,"opencv_contrib245.lib")
#pragma comment(lib,"opencv_legacy245.lib")
#pragma comment(lib,"opencv_flann245.lib")
#endif

static void help()
{
	printf("\nThis program demonstrated the use of the discrete Fourier transform (dft)\n"
		"The dft of an image is taken and it's power spectrum is displayed.\n"
		"Usage:\n"
		"./dft [image_name -- default lena.jpg]\n");
}

const char *keys =
{
	"{1| |lena.jpg|input image file}"
};


void convolveDFT(Mat A, Mat B, Mat C)
{
	
	// reallocate the output array if needed
	C.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());
	Size dftSize;
	// calculate the size of DFT transform
	dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
	dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

	// allocate temporary buffers and initialize them with 0's
	Mat tempA(dftSize, A.type(), Scalar::all(0));
	Mat tempB(dftSize, B.type(), Scalar::all(0));

	// copy A and B to the top-left corners of tempA and tempB, respectively
	Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
	A.copyTo(roiA);
	Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
	B.copyTo(roiB);

	// now transform the padded A & B in-place;
	// use "nonzeroRows" hint for faster processing
	dft(tempA, tempA, 0, A.rows);
	dft(tempB, tempB, 0, B.rows);

	// multiply the spectrums;
	// the function handles packed spectrum representations well
	mulSpectrums(tempA, tempB, tempA,DFT_ROWS);

	// transform the product back from the frequency domain.
	// Even though all the result rows will be non-zero,
	// you need only the first C.rows of them, and thus you
	// pass nonzeroRows == C.rows
	dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

	// now copy the result back to C.
	tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);

	// all the temporary buffers will be deallocated automatically
}


int main(int argc, const char **argv)
{
	help();                                                                                                                                                                                                                                     
	CommandLineParser parser(argc, argv, keys);
	string filename = parser.get<string>("1");

	Mat img = imread(filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	if( img.empty() )
	{
		help();
		printf("Cannot read image file: %s\n", filename.c_str());
		return -1;
	}
	Mat c;
	convolveDFT(img,img,c);
	int M = getOptimalDFTSize( img.rows );
	int N = getOptimalDFTSize( img.cols );
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	imshow("padded",padded);
	waitKey(1);
	Size nSize = padded.size();
	Mat planes[] = {Mat_<float>(padded), Mat::zeros(nSize, CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	// compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat mag = planes[0];
	mag += Scalar::all(1);
	log(mag, mag);

	// crop the spectrum, if it has an odd number of rows or columns
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(mag, mag, 0, 1, CV_MINMAX);

	imshow("spectrum magnitude", mag);
	waitKey();
	return 0;
}

