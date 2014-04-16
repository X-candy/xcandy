#include "LEDFlash.h"


CLEDFlash::CLEDFlash(void)
{

}

CLEDFlash::~CLEDFlash(void)
{
}


void CLEDFlash::operator()(Mat _img)
{
	
}

int CLEDFlash::DFTTransform(Mat _img,Mat &_mag)
{
	if(_img.channels()!=1 && !_img.empty())
		cvtColor(_img,_img,CV_RGB2GRAY);
	int M = getOptimalDFTSize( _img.rows );
	int N = getOptimalDFTSize( _img.cols );
	Mat padded;
	copyMakeBorder(_img, padded, 0, M - _img.rows, 0, N - _img.cols, BORDER_CONSTANT, Scalar::all(0));

#if LEDF_DEBUG
	imshow("_img",_img);
	imshow("padded",padded);
	waitKey(1);
	padded.convertTo(padded,CV_32F);
#endif
	
	Size nSize = padded.size();
	Mat planes[] = {Mat_<float>(padded), Mat::zeros(nSize, CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	// compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes[0]);
	_mag = planes[0];
	_mag += Scalar::all(1);
	log(_mag, _mag);

	// crop the spectrum, if it has an odd number of rows or columns
	_mag = _mag(Rect(0, 0, _mag.cols & -2, _mag.rows & -2));

	int cx = _mag.cols / 2;
	int cy = _mag.rows / 2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center
	Mat tmp;
	Mat q0(_mag, Rect(0, 0, cx, cy));
	Mat q1(_mag, Rect(cx, 0, cx, cy));
	Mat q2(_mag, Rect(0, cy, cx, cy));
	Mat q3(_mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(_mag, _mag, 0, 1, CV_MINMAX);
	imshow("spectrum magnitude", _mag);

	Mat c;
	convolveDFT(_img,_img,c);
	waitKey();
	return 1;
}

int CLEDFlash::convolveDFT(Mat _A, Mat _B, Mat &_C)
{
	/*if(_A.channels()!=1)
		cvtColor(_A,_A,CV_RGB2GRAY);

	if(_B.channels()!=1)
		cvtColor(_B,_B,CV_RGB2GRAY);*/

	// reallocate the output array if needed
	_C.create(abs(_A.rows - _B.rows)+1, abs(_A.cols - _B.cols)+1, _A.type());
	Size dftSize;
	// calculate the size of DFT transform
	dftSize.width = getOptimalDFTSize(_A.cols + _B.cols - 1);
	dftSize.height = getOptimalDFTSize(_A.rows + _B.rows - 1);

	// allocate temporary buffers and initialize them with 0's
	Mat tempA(dftSize, _A.type(), Scalar::all(0));
	Mat tempB(dftSize, _B.type(), Scalar::all(0));

	// copy A and B to the top-left corners of tempA and tempB, respectively
	Mat roiA(tempA, Rect(0,0,_A.cols,_A.rows));
	_A.copyTo(roiA);
	Mat roiB(tempB, Rect(0,0,_B.cols,_B.rows));
	_B.copyTo(roiB);

	imshow("tempA",tempA);
	waitKey(1);
	// now transform the padded A & B in-place;
	// use "nonzeroRows" hint for faster processing
	dft(tempA, tempA, 0, _A.rows);
	dft(tempB, tempB, 0, _B.rows);

	// multiply the spectrums;
	// the function handles packed spectrum representations well
	mulSpectrums(tempA, tempB, tempA,DFT_ROWS);

	// transform the product back from the frequency domain.
	// Even though all the result rows will be non-zero,
	// you need only the first C.rows of them, and thus you
	// pass nonzeroRows == C.rows
	dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, _C.rows);

	// now copy the result back to C.
	tempA(Rect(0, 0, _C.cols, _C.rows)).copyTo(_C);

	// all the temporary buffers will be deallocated automatically
	return 1;
}