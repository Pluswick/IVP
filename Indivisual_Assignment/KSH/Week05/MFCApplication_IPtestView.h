
// MFCApplication_IPtestView.h: CMFCApplicationIPtestView 클래스의 인터페이스
//

#pragma once


class CMFCApplicationIPtestView : public CView
{
protected: // serialization에서만 만들어집니다.
	CMFCApplicationIPtestView() noexcept;
	DECLARE_DYNCREATE(CMFCApplicationIPtestView)

// 특성입니다.
public:
	CMFCApplicationIPtestDoc* GetDocument() const;

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual void OnDraw(CDC* pDC);  // 이 뷰를 그리기 위해 재정의되었습니다.
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// 구현입니다.
public:
	virtual ~CMFCApplicationIPtestView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnDownSampling();
	afx_msg void OnUpSampling();
	afx_msg void OnSumConstant();
	afx_msg void OnSubConstant();
	afx_msg void OnMulConstant();
	afx_msg void OnDivConstant();
	afx_msg void OnAndOperate();
	afx_msg void OnOrOperate();
	afx_msg void OnXorOperate();
	afx_msg void OnNegaTransform();
	afx_msg void OnGammaCorrection();
	afx_msg void OnBinarization();
	afx_msg void OnStressTransform();
};

#ifndef _DEBUG  // MFCApplication_IPtestView.cpp의 디버그 버전
inline CMFCApplicationIPtestDoc* CMFCApplicationIPtestView::GetDocument() const
   { return reinterpret_cast<CMFCApplicationIPtestDoc*>(m_pDocument); }
#endif

