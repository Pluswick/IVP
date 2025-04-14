// ConstantDlg.cpp: 구현 파일
//

#include "pch.h"
#include "MFCApplication_IPtest.h"
#include "afxdialogex.h"
#include "ConstantDlg.h"


// ConstantDlg 대화 상자

IMPLEMENT_DYNAMIC(ConstantDlg, CDialog)

ConstantDlg::ConstantDlg(CWnd* pParent /*=nullptr*/)
	: CDialog(IDD_DIALOG4, pParent)
	, m_StartPoint(0)
	, m_EndPoint(0)
{

}

ConstantDlg::~ConstantDlg()
{
}

void ConstantDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, m_StartPoint);
	DDV_MinMaxInt(pDX, m_StartPoint, 0, 255);
	DDX_Text(pDX, IDC_EDIT2, m_EndPoint);
	DDV_MinMaxInt(pDX, m_EndPoint, 0, 255);
}


BEGIN_MESSAGE_MAP(ConstantDlg, CDialog)
END_MESSAGE_MAP()


// ConstantDlg 메시지 처리기
