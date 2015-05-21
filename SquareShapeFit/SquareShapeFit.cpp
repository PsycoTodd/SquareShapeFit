// SquareShapeFit.cpp : Defines the entry point for the console application.
//

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
RowVectorXd g_MeanShape;
RowVectorXd g_EigenVec;
CImg<unsigned char> g_pImg;
RowVectorXd g_EdgeHisto;

//Pre-define function

// Try to convert the shape space coordinate to the joint square space. This can be done due to that they share the same center.
void covertoJointSpace(float x, float y, int joint_x, int joint_y, int &jx, int &jy);

// This function tries to check the pixel value in the Iy at (x, y), which is in the joint space.
// In this way we don't need to waste the memory to creat space to really save the curShape pixel in every iteration
int getRegionAna(int x, int y, int joint_x, int joint_y, RowVectorXd &curShape, bool constructBin);

// This is the function to read the binary result from target image buffer at (x, y), which is in the joint space.
int getRegionImg(int x, int y, int joint_x, int joint_y, CImg<unsigned char> &img);

// This is a more general function also return sign if the pixel is on the edge of polygon (so we can easily take it into account for gradient calculation).
int insidePolygonAndOnEdge(float x, float y, RowVectorXd &curShape);

// We test the histogram mechanism here by get the nearest control point index(base 0)
unsigned int getBinNumber(float sx, float y, RowVectorXd &curShape);


static lbfgsfloatval_t evaluate0(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    int i;
    lbfgsfloatval_t fx = 0.0;
    //for (i = 0;i < n;i += 2) {
    //    lbfgsfloatval_t t1 = 1.0 - x[i];
    //    lbfgsfloatval_t t2 = 10.0 * (x[i+1] - x[i] * x[i]);
    //    g[i+1] = 20.0 * t2;
    //    g[i] = -2.0 * (x[i] * g[i+1] + t1);
    //    fx += t1 * t1 + t2 * t2;
    //}
    //return fx;

	lbfgsfloatval_t t1 = x[0]*x[0];
	g[0] = 2*x[0];
	fx = t1;
	return fx;
}



static lbfgsfloatval_t evaluate(
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
	g_EdgeHisto.setZero();
	//deform the shape to current shape
	//cout<<g_MeanShape.cols()<<' '<<g_EigenVec.cols();
	RowVectorXd curShape = g_MeanShape + g_EigenVec * x[0];
	//Now we can make the joint region
	//int cx = 32, cy = 32; //should be set by a claster center search result of the target outside of this func
	int joint_x = max(g_pImg.width()*1.0, curShape(0,0)-curShape(0,8));
	int joint_y = max(g_pImg.height()*1.0, curShape(0,9)-curShape(0,17));// joint size should be just a simple bounding box
	for(int i=0; i<joint_y; ++i)
	{
		for(int j=0; j<joint_x; ++j)
		{
			int spix = getRegionAna(j, i, joint_x, joint_y, curShape, false);
			int tpix = getRegionImg(j, i, joint_x, joint_y, g_pImg);
			fx += (spix-tpix) * (spix-tpix);
			g[0] += (spix-tpix);
			// Now the idea is simple, we try to count how many pixels on the edge of the shape model, put them into a histogram bin
			// Then in the following calculation, we just times the number of the count to the corresponding control point, i.e. we use
			// the nearest neightbor way to sample the gradient since the shape change need to be part of the derivitive or else the lbfgs cannot
			// figure out the acceleration change then it cannot find the minimum.
/*			float oneTerm = 2*(spix-tpix);
			//Now we try to get the gradient right
			//We need to know pixel value in the shape space: (x, y+1), (x+1, y), (x+1, y+1)
			int spix1 = getRegionAna(j-1, i, joint_x, joint_y, curShape, false);
			int spix2 = getRegionAna(j+1, i, joint_x, joint_y, curShape, false);
			int spix3 = getRegionAna(j, i+1, joint_x, joint_y, curShape, false);
			int spix4 = getRegionAna(j, i-1, joint_x, joint_y, curShape, false);
			//secTermX = abs(0.5*((spix2-1) + (spix3-spix1)));
			//secTermY = abs(0.5*((spix1 - 1) + (spix3-spix2)));
			float secTermX = 0.5*(spix2-spix1);
			float secTermY = 0.5*(spix3-spix4);
			g[0] += oneTerm;//*(secTermX+secTermY);*/
		}
	}
	//cout<<"G: "<<g[0]<<endl;

	//cout<<g_EdgeHisto<<endl;
	////now we try to summarize the gradient term
	////actually, the gradient term only may not be zero at the control point location

	//int controlPtNum = curShape.cols() / 2;
	//float oneTerm, secTermX, secTermY;
	//for(int i=0; i<controlPtNum; ++i)
	//{
	//	//now when we know the accordinate in the shape space, how can we convert it into the joint space?
	//	float sx = curShape(0, 2*i); 
	//	float sy = curShape(0, 2*i+1); //now we have the coordinate in the shape space
	//	int jx, jy;
	//	covertoJointSpace(sx, sy, joint_x, joint_y, jx, jy);
	//	int tpix = getRegionImg(jx, jy, joint_x, joint_y, g_pImg);
	//	if(tpix==1)
	//		//continue; //since this is the control point, the gradient is definitely 1, if the tpix is also one, then the sub is 0,
	//		// In this case, our bondary is inside the target space, we try to extend the shape
	//		oneTerm = -1;
	//	else
	//		oneTerm = 1;
	//	//Now we try to get the gradient right
	//	//We need to know pixel value in the shape space: (x, y+1), (x+1, y), (x+1, y+1)
	//	int spix1 = getRegionAna(jx-1, jy, joint_x, joint_y, curShape, false);
	//	int spix2 = getRegionAna(jx+1, jy, joint_x, joint_y, curShape, false);
	//	int spix3 = getRegionAna(jx, jy+1, joint_x, joint_y, curShape, false);
	//	int spix4 = getRegionAna(jx, jy-1, joint_x, joint_y, curShape, false);
	//	//secTermX = abs(0.5*((spix2-1) + (spix3-spix1)));
	//	//secTermY = abs(0.5*((spix1 - 1) + (spix3-spix2)));
	//	secTermX = 0.5*(spix2-spix1);
	//	secTermY = 0.5*(spix3-spix4);
	//	
	//	g[0] += 2*oneTerm*(secTermX*g_EigenVec(0,2*i) + secTermY*g_EigenVec(0,2*i+1))*g_EdgeHisto(0, i);//-2*1*1*(g_EigenVec(0, 2*i) + g_EigenVec(0, 2*i+1));
	//}
	//cout<<"G: "<<g[0]<<endl;
	return fx;


}

MatrixXf
	calcSobel(const MatrixXf &im, bool horizontal)
{
	Matrix3f sobel_kernel;
	Matrix3f &sk = sobel_kernel;//easy access
	int ks = sk.rows(); //get kernel size
	int hks = ks/2;
	if(!horizontal){
		sk<<-1,-2,-1,
			0,0,0,
			1,2,1;
	}
	else{
		sk<<-1,0,1,
			-2,0,2,
			-1,0,1;
	}
	float val;
	int imr = im.rows();
	int imc = im.cols();
	MatrixXf ret(imr, imc);
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

// we create the vector contain the mapping between the coordinate in the joint
// image with the corresponding eigen weight so later we can bayricentric the pixel as we wish
void buildWeightVector(const RowVectorXd &curShape, map<int, float> &xv,
						map<int, float> &yv, const RowVectorXd &eigenWei, int joint_x, int joint_y)
{
	int ptNum = curShape.cols()/2;
	for(int i=0; i<ptNum; ++i){
		float sx = curShape(0,2*i);
		float sy = curShape(0,2*i+1);
		int jx, jy;
		covertoJointSpace(sx, sy, joint_x, joint_y, jx, jy);
		float xWei = eigenWei(0, 2*i);
		float yWei = eigenWei(0, 2*i+1);
		xv.insert(pair<int, float>(jx, xWei));
		yv.insert(pair<int, float>(jy, yWei));
	}
	return;
}

float getEigenWeight(int cor, const map<int, float> &vecList)
{
	auto itlow = vecList.lower_bound(cor);
	auto itup = vecList.upper_bound(cor);
	if(itlow==vecList.end() || itup==vecList.end() || itup->first == itlow->first)
		return 0;
	int lowfactor = (cor - itlow->first)/(itup->first - itlow->first);
	int upfactor = (itup->first - cor)/(itup->first - itlow->first);
	return itlow->second * upfactor + itup->second * lowfactor;
}


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
	RowVectorXd curShape = g_MeanShape + g_EigenVec * x[0];
	//Now we can make the joint region
	//int cx = 32, cy = 32; //should be set by a claster center search result of the target outside of this func
	int joint_x = max(g_pImg.width()*1.0, curShape(0,0)-curShape(0,8))+0.5;
	int joint_y = max(g_pImg.height()*1.0, curShape(0,9)-curShape(0,17))+0.5;// joint size should be just a simple bounding box
	//Now we need to synthesis the Iy image buffer
	MatrixXf currentImg(joint_y, joint_x);
	currentImg.setZero();
	int jbx, jby, jex, jey;
	covertoJointSpace(curShape(0,8), curShape(0,9), joint_x, joint_y, jbx, jby);
	covertoJointSpace(curShape(0,24), curShape(0,25), joint_x, joint_y, jex, jey);
	currentImg.block(jby, jbx, jey-jby+1, jex-jbx+1).setOnes(); //now we set the center as a white pixel image
	MatrixXf vSobelRes = calcSobel(currentImg, false);
//	cout<<vSobelRes<<endl;
	MatrixXf hSobelRes = calcSobel(currentImg, true);
//	cout<<hSobelRes.count()<<endl;
	//Also try to build the map for control point and corresponding weight
	map<int, float> map_ControlPtX, map_ControlPtY;
	buildWeightVector(curShape, map_ControlPtX, map_ControlPtY, g_EigenVec, joint_x, joint_y);

	float T1, T2x, T2y, T3x, T3y; //several different term
	for(int i=0; i<joint_y; ++i)
	{
		for(int j=0; j<joint_x; ++j)
		{
			T3x = T3y = 0;
			int tpix = getRegionImg(j, i, joint_x, joint_y, g_pImg);
			int spix = currentImg(i, j);
			fx += (spix-tpix) * (spix-tpix);
			T1 = 2.0*(spix-tpix)/fstd;
			T2y = abs(vSobelRes(i, j));
			T2x = abs(hSobelRes(i, j));
			//For T3, we only care about the point exist in the sobel since that is the only way the multiplication can survive.
			if(T2y && T1){

				T3x = abs(getEigenWeight(j, map_ControlPtX));
				//cout<<i<<' '<<j<<' '<<T3x<<endl;
			}
			if(T2x && T1){
				T3y = abs(getEigenWeight(i, map_ControlPtY));
				//cout<<i<<' '<<j<<' '<<T3y<<endl;
			}
			g[0] += T1*(T2x*T3x+T2y*T3y);
			
		}

	}
	cout<<"G: "<<g[0]<<endl;
	return fx;
}



static int progress0(
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
    printf("  fx = %f, x[0] = %f\n", fx, x[0]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
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

bool insidePolygon(float x, float y, RowVectorXd &curShape)
{
	//use cross product way to get the correct result
	// This is because we know that the center of the shape is 0
	int ptNum = curShape.cols()/2;
	RowVector3d p1, p2, org, tar;
	org(0,0) = 0; org(0,1) = 0; org(0,2) = 0;
	tar(0,0) = x; tar(0,1) = y; tar(0,2) = 0;
	p1(0,2) = p2(0,2) = 0;
	RowVector3d cr1, cr2; //two cross product
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

int insidePolygonAndOnEdge(float x, float y, RowVectorXd &curShape)
{
	// If it is inside, return 1, outside return 0 on Edge(in a small range) return -1
	int condition;
	bool onEdge = false;
	const float insideCondition = 0.05;
	int ptNum = curShape.cols()/2;
	RowVector3d p1, p2, org, tar, temp;
	org(0,0) = 0; org(0,1) = 0; org(0,2) = 0;
	tar(0,0) = x; tar(0,1) = y; tar(0,2) = 0;
	p1(0,2) = p2(0,2) = 0;
	float cre, e1, e2;
	RowVector3d cr1, cr2; //two cross products
	for(int i=0; i<ptNum; ++i){
		int j = (i+1)%ptNum;
		p1(0,0) = curShape(0,2*i);
		p1(0,1) = curShape(0,2*i+1);
		p2(0,0) = curShape(0,2*j);
		p2(0,1) = curShape(0,2*j+1);
		//get the default cross product
		cr1 = (p2-p1).cross(org-p1);
		cr2 = (p2-p1).cross(tar-p1);
		temp = p2-p1;
		e1 = sqrt(temp.x()*temp.x() + temp.y()*temp.y());
		temp = tar-p1;
		e2 = sqrt(temp.x()*temp.x() + temp.y()*temp.y());
		cre = cr2.z(); //quad_size
		cre = cre/(e1*e2); //sin
		if(cr1.dot(cr2)<0){
			condition = 0;
			//if(abs(cre)<insideCondition)
			//	condition = -1; //on the edge we think
			break;
		}
		else{
			condition = 1;
			if(abs(cre)<insideCondition)
			{
				onEdge = true;
			}
		}
		//now we see the area size of cross, if it is close to 0, we have point on line
	}
	if(condition==0)
		return condition;
	else if(onEdge)
		return -1;
	else
		return 1;
}

unsigned int getBinNumber(float sx, float sy, RowVectorXd &curShape)
{
	int ptNum = curShape.cols()/2;
	float minDis = INT_MAX, minInd;
	float cx, cy, cd;
	for(unsigned int i=0; i<ptNum; i++)
	{
		cx = curShape(0,2*i);
		cy = curShape(0,2*i+1);
		cd = (cx-sx)*(cx-sx) + (cy-sy)*(cy-sy);
		if(cd<minDis){
			minDis = cd;
			minInd = i;
		}
	}
	return minInd;
}

int getRegionAna(int x, int y, int joint_x, int joint_y, RowVectorXd &curShape, bool constructBin)
{
	if(x<0 || x>=joint_x || y<0 || y>=joint_y) return 0;
	float jcx = (joint_x-1)/2.0;
	float jcy = (joint_y-1)/2.0;
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
	float jcx = (joint_x-1)/2.0;
	float jcy = (joint_y-1)/2.0;
	int w = img.width();
	int h = img.height();
	float icx = (w-1)/2.0;
	float icy = (h-1)/2.0;
	float iox = jcx-icx;
	float ioy = jcy-icy; //now we know in the joint space, where the origin of the image is.
	int ix = static_cast<int>(x - iox + 0.5);
	int iy = static_cast<int>(y - ioy + 0.5);
	if(ix<0 || iy<0 || ix>=w || iy>=h)
		return 0;
	else
		return img(ix, iy, 0, 1)/255;
}


float
	SquareImgFit(CShapeExpert &sm, CImg<unsigned char> &pImg)
{
	// The main entrance we use to start our work
	float p;
	//Basic idea:
/*	1. we need to find the center of the target image, then we move the shape model's
	center to there, and create the joint region image based on the coverage
	2. The same range R, we have different mask, so we have values in each pixel.
	3. To make it like a 2D function, we will put x=pixel index, and y=0/1 depending on the coverage
	4. In this case, the gradient is just a certain 1 at certain x indeices, which do not need sobel at all
	in our case since we know the contour.
	5. But the result is more complicated since the x here is not the variable we can control, shape weight is.
	6. The shape weight's derivitive, actually is the eigenvector, and our shape weight factor is the x.
	In this case, the gridient term is very simple since the contributors are just the 16 control points we can manipulate with.
	Yeah, we still calculate the entire joint image, the entire edge for image processing purpose because it is more easy, but finally
	most of the term then it comes to gridient is zero except the 16 control point that we can control. However, once the shape weight
	changes, the function value will change. 16 variables here are enough.
	
*/


	return p;
}

int _tmain(int argc, _TCHAR* argv[])
{
	string testImgPath(argv[1]);
	int imageNum = 20;
	char imgName[256];
	vector<float> wvector;
	vector<bool> falseConverge;
	vector<double> timevec;
	CShapeExpert sm("G:/XING/3DCLMTracking/lbfgsTest/SquareShapeFit/SquareShapeFit/squareShapeModel.txt", 16, 2); // We load the shape model here

	lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(1);
    lbfgs_parameter_t param;

	 if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return 1;
    }

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
	MatrixXd ev = sm.getEigenVector(); //our optimized variable will be the weight for each vector
	g_EigenVec = ev.transpose();
	int ptNum = g_MeanShape.cols()/2;
	g_EdgeHisto = RowVectorXd(ptNum);
	std::clock_t start;
	//*x = 5;
	for(int i=0; i<imageNum; ++i)
	{
		sprintf(imgName, "%s%d.bmp", testImgPath.c_str(), i+1);
		g_pImg.load(imgName);
		g_EdgeHisto.setZero();
		*x =5;
		lbfgs_parameter_init(&param);

		start = clock();
		ret = lbfgs(1, x, &fx, evaluateImg, progress, NULL, &param); //evaluate works for inital x=5 for some case
		timevec.push_back((clock()-start)/(CLOCKS_PER_SEC/1000));
		if(ret)
		{
			cout<<"LBFGS converge error code: "<<i<<' '<<ret<<endl;
			falseConverge.push_back(true);
		}
		else
			falseConverge.push_back(false);

		wvector.push_back(*x);
	}
	for(auto a=wvector.begin(); a<wvector.end(); ++a)
	{
		RowVectorXd curShape = g_MeanShape + g_EigenVec * (*a);
		float edge = curShape(0,0) - curShape(0,8);
		if(falseConverge[a-wvector.begin()]==true)
			printf("*  "); 
		printf("%f   %f\n", edge, timevec[a-wvector.begin()]);

	}
	lbfgs_free(x);
	return 0;
}


//#define N 1
//
//int _tmain(int argc, _TCHAR* argv[])
//{
//	int i, ret = 0;
//	lbfgsfloatval_t fx;
//    lbfgsfloatval_t *x = lbfgs_malloc(N);
//    lbfgs_parameter_t param;
//
//    if (x == NULL) {
//        printf("ERROR: Failed to allocate a memory block for variables.\n");
//        return 1;
//    }
//
//    /* Initialize the variables. */
//    for (i = 0;i < N;i ++) {
//        x[i] = 5.2;
//    }
//
//    /* Initialize the parameters for the L-BFGS optimization. */
//    lbfgs_parameter_init(&param);
//    /*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/
//
//    /*
//        Start the L-BFGS optimization; this will invoke the callback functions
//        evaluate() and progress() when necessary.
//     */
//    ret = lbfgs(N, x, &fx, evaluate0, progress0, NULL, &param);
//
//    /* Report the result. */
//    printf("L-BFGS optimization terminated with status code = %d\n", ret);
//    printf("  fx = %f, x[0] = %f\n", fx, x[0]);
//
//    lbfgs_free(x);
//    return 0;
//}

