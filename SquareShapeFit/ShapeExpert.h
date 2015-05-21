#pragma once
#include <Dense>
#include <vector>

using namespace Eigen;
class CShapeExpert
{
public:
	CShapeExpert(const char *shapeModelFName, size_t pointNum, size_t dim);
	virtual ~CShapeExpert(void);
	RowVectorXf &getMeanShape() {return *m_vMeanShape;};
	MatrixXf &getEigenVector() {return *m_mEigenVec;};
private:
	void loadModelFromFile(const char *shapeModelFName);
	
private:
	MatrixXf *m_mEigenVec;
	VectorXf *m_vEigenVar;
	MatrixXf *m_mMeanShape; //Standard mean shape storage, each row is a vetex
	RowVectorXf *m_vMeanShape;
	size_t m_uPointNum;
	size_t m_uDim;
	//temp var for arranging the data
	std::vector<float> m_vMS, m_vEVar, m_vEVec;

};

