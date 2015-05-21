#include "StdAfx.h"
#include "ShapeExpert.h"
#include <fstream>
#define OFFSET 0

CShapeExpert::CShapeExpert(const char *shapeModelFName, size_t pointNum, size_t dim)
{
	m_uPointNum = pointNum;
	m_uDim = dim;
	loadModelFromFile(shapeModelFName);
	//now we have sth in the temp vectors
	size_t eigenVarNum = m_vEVar.size();
	m_mEigenVec = new MatrixXf(m_uPointNum*m_uDim, eigenVarNum);
	m_vEigenVar = new VectorXf(eigenVarNum);
	m_mMeanShape = new MatrixXf(m_uPointNum, m_uDim+OFFSET);
	m_vMeanShape = new RowVectorXf(m_uPointNum*(m_uDim+OFFSET));
	//load the data
	size_t ind = 0, row, col;
	for(std::vector<float>::iterator it=m_vMS.begin(); it<m_vMS.end(); ++it)
	{
		row = ind/(m_uDim+OFFSET);
		col = ind%(m_uDim+OFFSET);
		(*m_mMeanShape)(row, col) = *it;
		(*m_vMeanShape)(ind++) = *it;
	}
	ind = 0;
	for(std::vector<float>::iterator it=m_vEVar.begin(); it<m_vEVar.end(); ++it)
		(*m_vEigenVar)(ind++) = *it;
	ind = 0;
	for(std::vector<float>::iterator it=m_vEVec.begin(); it<m_vEVec.end(); ++it, ++ind)
	{
		row = ind/eigenVarNum;
		col = ind%eigenVarNum;
		(*m_mEigenVec)(row, col) = *it;
	}

	//clear the temp vector
	m_vMS.clear();
	m_vEVar.clear();
	m_vEVec.clear();
}


CShapeExpert::~CShapeExpert(void)
{
	if(m_mEigenVec)
	{
		delete m_mEigenVec;
		m_mEigenVec = NULL;
	}
	if(m_vEigenVar)
	{
		delete m_vEigenVar;
		m_vEigenVar = NULL;
	}
	if(m_mMeanShape)
	{
		delete m_mMeanShape;
		m_mMeanShape = NULL;
	}
	if(m_vMeanShape)
	{
		delete m_vMeanShape;
		m_vMeanShape = NULL;
	}

}

void
	CShapeExpert::loadModelFromFile(const char *shapeModelFName)
{
	float fbuf;
	char cbuf;
	int count = 0;
	std::vector<float> *p = &m_vMS;
	std::fstream fid(shapeModelFName);
	while(fid.good())
	{
		cbuf = fid.peek();
		if(cbuf == '\n' || cbuf == ' ')
		{
			fid.get(cbuf);
			continue;
		}
		if(cbuf == '*')
		{
			fid.get(cbuf);
			if(count == 0)
			{
				p = &m_vEVar;
			}
			if(count == 1)
			{
				p = &m_vEVec;
			}
			
			count++;
			continue;
		}
		fid>>fbuf;
		p->push_back(fbuf);
	}
	p->pop_back();
	fid.close();
}
