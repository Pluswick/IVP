
// MFCApplication_IPtest.h: MFCApplication_IPtest ���ø����̼��� �⺻ ��� ����
//
#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'pch.h'�� �����մϴ�."
#endif

#include "resource.h"       // �� ��ȣ�Դϴ�.


// CMFCApplicationIPtestApp:
// �� Ŭ������ ������ ���ؼ��� MFCApplication_IPtest.cpp��(��) �����ϼ���.
//

class CMFCApplicationIPtestApp : public CWinAppEx
{
public:
	CMFCApplicationIPtestApp() noexcept;


// �������Դϴ�.
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// �����Դϴ�.
	UINT  m_nAppLook;
	BOOL  m_bHiColorIcons;

	virtual void PreLoadState();
	virtual void LoadCustomState();
	virtual void SaveCustomState();

	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CMFCApplicationIPtestApp theApp;
