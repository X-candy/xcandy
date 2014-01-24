#include "common.h"
#include <math.h>

#ifdef _FPCLASS_SNAN
#include <math.h>
#define NaN sqrt(-1.0)
#define IsNaN(x) _isnan(x)
#else
#define NaN (0.0 / 0.0)
#define IsNaN(x) ((x) != (x))
#endif

#ifdef _FPCLASS_NINF
#include <math.h>
#define INF DBL_MAX
#define PINF DBL_MAX
#define NINF -DBL_MAX
#define IsINF(x) _finite(x)
#else
#define INF	 -log(0) //无穷大
#define PINF	 INF	 //正无穷大
#define NINF	 -INF	 //负无穷大
#define isINF(x)	 (((x)==PINF)||((x)==NINF))
#endif


struct HSRect
{
	float x;
	float y;
	float width;
	float height;
	float center_point[2];
	HSRect()
	{
		x=0;
		y=0;
		width=0;
		height=0;
		memset(center_point,0,sizeof(float)*2);
	}
};


struct DATASET
{
	char name[255];
	int id;
	int t_start;
	int t_end;
	int length;
	vector<int> omega;
	vector<HSRect> rect;
	DATASET()
	{
		memset(name,0,255);
		t_start=0;
		t_end = 0;
		length=0;
		omega.clear();
		rect.clear();
	}
};

struct DETECTRECT
{
	vector<int> idx;
	vector<HSRect> detect_rect;
	DETECTRECT()
	{
		idx.clear();
		detect_rect.clear();
	}
};


struct I_TRACK_LINK
{
	int id;
	int t_start;
	int t_end;
	int length;
	Mat omega;
	Mat xy_data;
	double rank;
	I_TRACK_LINK()
	{
		id =0;
		t_start=0;
		t_end=0;
		length=0;
		omega=Mat();
		xy_data=Mat();
		rank=NaN;
	}
};

struct RESULTS
{
	double  rank;
	double max_rank;
	double min_rank;
	BOOL qcheck;
	double gap;
	double lambda;

	RESULTS()
	{
		rank=NaN;
		qcheck=0;
		max_rank=NaN;
		min_rank=NaN;
		gap = NaN;
		lambda=0.1;
	}
};



int read_dataset(char* _file_path,vector<DATASET> &_dataset);
int ProcessDataSet(vector<DETECTRECT> &_detect_rect,vector<DATASET> &_dataset);
int findAssociations(vector<DETECTRECT> &_detect_rect,int _ratio_threhold,vector<Mat> &_b,vector<Mat> &_distance);
int linkDetectionTracklets(vector<DETECTRECT> &_detect_rect,vector<Mat> _b,vector<Mat> _distance,vector<I_TRACK_LINK> &_itl);
int initial_track();

void DiffMat(Mat _a,Mat &_b);
void nonunique(Mat _a,Mat &_b);
int getxychain(vector<DETECTRECT> &_detect_rect,vector<Mat> &_b,int _frame,int _ind,Mat &_xy);
int growitl(vector<I_TRACK_LINK> _itl,int max_D);
int get_itl_horizon(vector<I_TRACK_LINK> _itl,int _t_start,int _t_end,vector<I_TRACK_LINK> &_itlh);
int associate_itl(vector<I_TRACK_LINK> _itl,int _t_start,int _t_end);