// CUpSampleDlg.cpp: 구현 파일
//

#include "pch.h"
#include "MFC_assignment04.h"
#include "afxdialogex.h"
#include "CUpSampleDlg.h"


// CUpSampleDlg 대화 상자

IMPLEMENT_DYNAMIC(CUpSampleDlg, CDialog)

CUpSampleDlg::CUpSampleDlg(CWnd* pParent /*=nullptr*/)
	: CDialog(IDD_DIALOG1, pParent)
	, m_UpSampleRate(0)
{

}

CUpSampleDlg::~CUpSampleDlg()
{
}

void CUpSampleDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, m_UpSampleRate);
	DDV_MinMaxInt(pDX, m_UpSampleRate, 1, 32);
}


BEGIN_MESSAGE_MAP(CUpSampleDlg, CDialog)
END_MESSAGE_MAP()


// CUpSampleDlg 메시지 처리기
