
struct NOISE	
{
	double fn;
	double fp;
	double gn;
	NOISE()
	{
		fn = 0;
		fp = 0;
		gn = 0;
	}
};

struct MOT
{
	double fn;
	double fp;
	double mme;
	double g;
	double mota;
	double mmerat;


	MOT()
	{
		fn =0;
		fp=0;
		mme=0;
		g=0;
		mota=0;
		mmerat=0;
	}
};

struct DEFAULT_PARAMS
{
	int similarity_method;
	float min_similarity;
	float mota_threhold;
	bool debug;
	int hor;
	float eta_max;
	BOOL qcheck;
	double gap;
	double hor_max ;
	double gap_max ;
	double slope_max;
	
	DEFAULT_PARAMS()
	{
		similarity_method=1;
		min_similarity=0.01;
		mota_threhold=0.5;
		hor = 80;
		eta_max = 3;
		debug=1;
		gap=0;
		qcheck=false;
		hor_max = INF;
		gap_max = INF;
		slope_max= INF;
	}
};




