// This is a C style test code to test lbfgs with more variation a.k.a more high energy EVal.
// In this experiment, we will test about how 16x16 control point, with aligned center to the binary
// map, could fit to arbitrary shape include rotation rectange. In theory, with all 16 control points
// free, we can fit to any shape, that is in theory how the formular shoule be able to work.
#include "stdafx.h"
#include "lbfgs.h"
#include "CImg.h" //for image-based loading and processing
#include "ShapeExpert.h"
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <map>
using namespace std;
using namespace cimg_library;
RowVectorXf g_MeanShape;
MatrixXf g_EigenVec;
CImg<unsigned char> g_pImg;
RowVectorXf g_EdgeHisto;


//Pre-define function

// Try to convert the shape space coordinate to the joint square space. This can be done due to that they share the same center.
void covertoJointSpace(float x, float y, int joint_x, int joint_y, int &jx, int &jy);

// This function tries to check the pixel value in the Iy at (x, y), which is in the joint space.
// In this way we don't need to waste the memory to creat space to really save the curShape pixel in every iteration
int getRegionAna(int x, int y, int joint_x, int joint_y, const RowVectorXf &curShape, bool constructBin);

// This is the function to read the binary result from target image buffer at (x, y), which is in the joint space.
int getRegionImg(int x, int y, int joint_x, int joint_y, CImg<unsigned char> &img);

// This is a more general function also return sign if the pixel is on the edge of polygon (so we can easily take it into account for gradient calculation).
bool insidePolygon(float x, float y, const RowVectorXf &curShape);

// This is the function we used to calculate the real sobel converluted image with sign
// Although currently I still not figure out why the sign is not useful.
MatrixXf calcSobel(const MatrixXf &im, bool horizontal);

// This is for the third derivative component with eigen vector wector
void buildWeightVector(const RowVectorXf &curShape, map<int, RowVectorXf> &xv,
						map<int, RowVectorXf> &yv, const MatrixXf &eigenWei, int joint_x, int joint_y);

// This function helps to figure out the interpolated weigth vector
RowVectorXf getEigenWeight(int cor, const map<int, RowVectorXf> &vecList, int weightDim);

// This function helps to figure out the horizontal size of rotated range (hope it is not larger than the target image...)
float getShapeRange(const RowVectorXf &shape, bool horizontal);

//just create the binary sample array with all pixel inside the shape 1, so no smooth at all as real image
void buildupShapeImage(MatrixXf &currentImg, const RowVectorXf &curShape);


void covertoJointSpace(float x, float y, int joint_x, int joint_y, int &jx, int &jy)
{
	float jcx = (joint_x-1)/2.0;
	float jcy = (joint_y-1)/2.0;
	if(x<0)
		jx = static_cast<int>(jcx + x + 1);
	else
		jx = static_cast<int>(jcx + x);
	if(y<0)
		jy = static_cast<int>(jcy - y);
	else
		jy = static_cast<int>(jcy -y + 1);
}


int getRegionAna(int x, int y, int joint_x, int joint_y, const RowVectorXf &curShape, bool constructBin)
{
	if(x<0 || x>=joint_x || y<0 || y>=joint_y) return 0;
	float jcx = (joint_x-1)/2.0f;
	float jcy = (joint_y-1)/2.0f;
	float sx = x - jcx;
	float sy = y - jcy; //now we have the coord in the shape space
	sy = -sy;
	//now we test whether a point inside a convex polygon
	bool isInside = insidePolygon(sx, sy, curShape);
	/*if(isInside){
		cout<<x<<' '<<y<<endl;
	}*/
	return isInside?1:0;
/*	int isInside = insidePolygonAndOnEdge(sx, sy, curShape);
	if(isInside==-1 && constructBin == true) // here we try to construct the bin
	{
	//	cout<<x<<' '<<y<<endl;
		int idx = getBinNumber(sx, sy, curShape);
		g_EdgeHisto(0, idx) += 1;
	}
	return abs(isInside);*/
}

int getRegionImg(int x, int y, int joint_x, int joint_y, CImg<unsigned char> &img)
{
	float jcx = (joint_x-1)/2.0f;
	float jcy = (joint_y-1)/2.0f;
	int w = img.width();
	int h = img.height();
	float icx = (w-1)/2.0f;
	float icy = (h-1)/2.0f;
	float iox = jcx-icx;
	float ioy = jcy-icy; //now we know in the joint space, where the origin of the image is.
	int ix = static_cast<int>(x - iox + 0.5);
	int iy = static_cast<int>(y - ioy + 0.5);
	if(ix<0 || iy<0 || ix>=w || iy>=h)
		return 0;
	else
		return img(ix, iy, 0, 1)/255;
}


bool insidePolygon(float x, float y, const RowVectorXf &curShape)
{
	//use cross product way to get the correct result
	// This is because we know that the center of the shape is 0
	int ptNum = curShape.cols()/2;
	RowVector3f p1, p2, org, tar;
	org(0,0) = 0; org(0,1) = 0; org(0,2) = 0;
	tar(0,0) = x; tar(0,1) = y; tar(0,2) = 0;
	p1(0,2) = p2(0,2) = 0;
	RowVector3f cr1, cr2; //two cross product
	for(int i=0; i<ptNum; ++i){
		int j = (i+1)%ptNum;
		p1(0,0) = curShape(0,2*i);
		p1(0,1) = curShape(0,2*i+1);
		p2(0,0) = curShape(0,2*j);
		p2(0,1) = curShape(0,2*j+1);
		//get the default cross product
		cr1 = (p2-p1).cross(org-p1);
		cr2 = (p2-p1).cross(tar-p1);
		if(cr1.dot(cr2)<0) //just prove that the evaluate dot on the same side as the origin
			return false;
	}
	return true;
}

MatrixXi
	calcSobel(const MatrixXi &im, bool horizontal)
{
	Matrix3f sobel_kernel;
	Matrix3f &sk = sobel_kernel;//easy access
	int ks = sk.rows(); //get kernel size
	int hks = ks/2;
	if(!horizontal){
		sk<<1,2,1,
			0,0,0,
			-1,-2,-1;
	}
	else{
		sk<<-1,0,1,
			-2,0,2,
			-1,0,1;
	}
	int val;
	int imr = im.rows();
	int imc = im.cols();
	MatrixXi ret(imr, imc);
	for(int i=0; i<imr; ++i){
		for(int j=0; j<imc; ++j){
			val = 0;
			for(int k1=-hks; k1<=hks; ++k1){
				if(i+k1<0 || i+k1>=imc)
					continue;
				for(int k2=-hks; k2<=hks; ++k2){
					if(j+k2<0 || j+k2>=imc)
						continue;
					val += sk(hks+k1, hks+k2) * im(i+k1, j+k2);
				}
			}
			ret(i, j) = val;
		}
	}
	return ret;
}

void buildWeightVector(const RowVectorXf &curShape, map<int, RowVectorXf> &xv,
						map<int, RowVectorXf> &yv, const MatrixXf &eigenWei, int joint_x, int joint_y)
{
	int ptNum = curShape.cols()/2;
	for(int i=0; i<ptNum; ++i){
		float sx = curShape(0,2*i);
		float sy = curShape(0,2*i+1);
		int jx, jy;
		covertoJointSpace(sx, sy, joint_x, joint_y, jx, jy);
		RowVectorXf xWei = eigenWei.row(2*i);
		RowVectorXf yWei = eigenWei.row(2*i+1);
		xv.insert(pair<int, RowVectorXf>(jx, xWei));
		yv.insert(pair<int, RowVectorXf>(jy, yWei));
	}
	return;
}

RowVectorXf getEigenWeight(int cor, const map<int, RowVectorXf> &vecList, int weightDim)
{

	RowVectorXf ret(weightDim);
	ret.setZero();
	auto itlow = vecList.lower_bound(cor);
	auto itup = vecList.upper_bound(cor);
	if(itlow==vecList.end() || itup==vecList.end() || itup->first == itlow->first)
		return ret;
	int lowfactor = (cor - itlow->first)/(itup->first - itlow->first);
	int upfactor = (itup->first - cor)/(itup->first - itlow->first);
	for(int i=0; i<weightDim; ++i)
	{
		ret(0,i) = -(itlow->second(0,i) * upfactor + itup->second(0,i) * lowfactor);
	}
	return ret;
}

float getShapeRange(const RowVectorXf &shape, bool horizontal)
{
	int initial = horizontal?0:1;
	int step = 2;
	float low = INT_MAX, high = INT_MIN;
	for(; initial<shape.cols(); initial += 2)
	{
		if(shape(0, initial)<low) low = shape(0, initial);
		if(shape(0, initial)>high) high = shape(0, initial);
	}
	return high-low;
}

void buildupShapeImage(MatrixXi &currentImg, const RowVectorXf &shape) // This is super slow
{
	int imw = currentImg.cols();
	int imh = currentImg.rows();
	for(int i=0; i<imh; ++i)
		for(int j=0; j<imw; ++j){
			currentImg(i, j) = getRegionAna(j,i,imw, imh, shape, false);
		//	if(currentImg(i,j)!= 0)
		//		cout<<i<<' '<<j<<endl;
		}
}


// Main Entrance of lbgfs function
static lbfgsfloatval_t evaluateImg(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
	)
{
	int i;
	lbfgsfloatval_t fx = 0.0;
	g[0] = 0;
	float fstd = 12.4097*12.4097;
	int eigenDim = 1;
	MatrixXf wx(eigenDim,1);
	for(i=0; i<eigenDim; ++i)
		wx(i,0) = x[i];
	RowVectorXf curShape = g_MeanShape + static_cast<RowVectorXf>((g_EigenVec * wx).transpose());
	//Now we can make the joint region
	//int cx = 32, cy = 32; //should be set by a claster center search result of the target outside of this func
//	int joint_x = max(g_pImg.width()*1.0f, getShapeRange(curShape, true))+0.5f;
//	int joint_y = max(g_pImg.height()*1.0f, getShapeRange(curShape, false))+0.5f;// joint size should be just a simple bounding box
	int joint_x = max(g_pImg.width()*1.0f, curShape(0,0)-curShape(0,8))+0.5f;
	int joint_y = max(g_pImg.height()*1.0f, curShape(0,9)-curShape(0,17))+0.5f;
	
	
	//Now we need to synthesis the Iy image buffer
	MatrixXi currentImg(joint_y, joint_x);
	currentImg.setZero();
/*	buildupShapeImage(currentImg, curShape);*/
//	cout<<currentImg.count();

	int jbx, jby, jex, jey;
	covertoJointSpace(curShape(0,8), curShape(0,9), joint_x, joint_y, jbx, jby);
	covertoJointSpace(curShape(0,24), curShape(0,25), joint_x, joint_y, jex, jey);
	currentImg.block(jby, jbx, jey-jby+1, jex-jbx+1).setOnes();

	MatrixXi vSobelRes = calcSobel(currentImg, false);
//	cout<<vSobelRes<<endl;
	MatrixXi hSobelRes = calcSobel(currentImg, true);
//	cout<<hSobelRes.count()<<endl;
	//Also try to build the map for control point and corresponding weight
	map<int, RowVectorXf> map_ControlPtX, map_ControlPtY;
	buildWeightVector(curShape, map_ControlPtX, map_ControlPtY, g_EigenVec, joint_x, joint_y);

	float T1, T2x, T2y; //several different term
	RowVectorXf T3x(eigenDim), T3y(eigenDim);
	for(int i=0; i<joint_y; ++i)
	{
		for(int j=0; j<joint_x; ++j)
		{
			T3x.setZero(); T3y.setZero();
			int tpix = getRegionImg(j, i, joint_x, joint_y, g_pImg);
			int spix = currentImg(i, j);
			fx += (spix-tpix) * (spix-tpix)/fstd;
			T1 = 2.0*(spix-tpix)/fstd;
			T2y = vSobelRes(i, j);
			T2x = hSobelRes(i, j);
			//For T3, we only care about the point exist in the sobel since that is the only way the multiplication can survive.
			if(T2x && T1){

				T3x = getEigenWeight(j, map_ControlPtX, eigenDim);
				//cout<<i<<' '<<j<<' '<<T3x<<endl;
			}
			if(T2y && T1){
				T3y = getEigenWeight(i, map_ControlPtY, eigenDim);
				//cout<<i<<' '<<j<<' '<<T3y<<endl;
			}
			for(int e=0; e<eigenDim; ++e)
				g[e] += T1*(T2x*T3x(0,e)+T2y*T3y(0,e));

			
		}

	}
	//cout<<"G: "<<g[0]<<endl;//' '<<g[1]<<' '<<g[2]<<' '<<g[3]<<endl;
	return fx;
}


static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f\n", fx, x[0]); //since in square case, we just have one variable
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
	return 0;
}



int _tmain(int argc, _TCHAR* argv[])
{
	string testImgPath(argv[1]);
	int imageNum = 21;
	char imgName[256];
	vector<vector<float>> wvector;
	vector<bool> falseConverge;
	vector<double> timevec;
	CShapeExpert sm("G:/XING/3DCLMTracking/lbfgsTest/SquareShapeFit/SquareShapeFit/squareShapeModel.txt", 16, 2); // We load the shape model here

/*	MatrixXf a(3,3);
	a<<0,0,0,
	   1,1,0,
	   1,1,0;
	cout<<a<<endl;
	Matrix3f b = calcSobel(a, false);
	Matrix3f c = calcSobel(a, true);
	cout<<b+c<<endl;*/

	int ret = 0;

	// First, we need to get the meanshape and eigenvector. So we can use our x, i.e. shape weight to adjust them in the lbfgs function
	g_MeanShape = sm.getMeanShape();
	MatrixXf ev = sm.getEigenVector(); //our optimized variable will be the weight for each vector
	int eigenDim = ev.cols();
	g_EigenVec = ev;
	int ptNum = g_MeanShape.cols()/2;
	//g_EdgeHisto = RowVectorXd(ptNum);


	lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(eigenDim);
    lbfgs_parameter_t param;
	if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return 1;
    }


	std::clock_t start;
	//*x = 5;
	for(int i=20; i<imageNum; ++i)
	{
		sprintf(imgName, "%s%d.bmp", testImgPath.c_str(), i+1);
		g_pImg.load(imgName);
		
		for(int e=0; e<eigenDim; ++e)
			x[e] = 5;

		lbfgs_parameter_init(&param);

		start = clock();
		ret = lbfgs(eigenDim, x, &fx, evaluateImg, progress, NULL, &param); //evaluate works for inital x=5 for some case
		timevec.push_back((clock()-start)/(CLOCKS_PER_SEC/1000));
		if(ret)
		{
			cout<<"LBFGS converge error code: "<<i<<' '<<ret<<endl;
			falseConverge.push_back(true);
		}
		else
			falseConverge.push_back(false);

		vector<float> member;
		for(int e=0; e<eigenDim; ++e){
			member.push_back(x[e]);
		}
		wvector.push_back(member);
	}
	for(auto a=wvector.begin(); a<wvector.end(); ++a)
	{
		
		MatrixXf wx(eigenDim,1);
		for(int i=0; i<eigenDim; ++i)
			wx(i,0) = (*a)[i];
		RowVectorXf curShape = g_MeanShape + static_cast<RowVectorXf>((g_EigenVec * wx).transpose());
		float edge = curShape(0,0) - curShape(0,8);
		if(falseConverge[a-wvector.begin()]==true)
			printf("*  "); 
		printf("%f   %f\n", edge, timevec[a-wvector.begin()]);

	}
	lbfgs_free(x);
	return 0;
}