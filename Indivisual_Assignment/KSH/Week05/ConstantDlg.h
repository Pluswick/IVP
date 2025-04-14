#pragma once
#include "afxdialogex.h"


// ConstantDlg 대화 상자

class ConstantDlg : public CDialog
{
	DECLARE_DYNAMIC(ConstantDlg)

public:
	ConstantDlg(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~ConstantDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG4 };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

	DECLARE_MESSAGE_MAP()
public:
	int m_StartPoint;
	int m_EndPoint;
};
