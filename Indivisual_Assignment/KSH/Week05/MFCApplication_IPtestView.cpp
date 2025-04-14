
// MFCApplication_IPtestView.cpp: CMFCApplicationIPtestView 클래스의 구현
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "MFCApplication_IPtest.h"
#endif

#include "MFCApplication_IPtestDoc.h"
#include "MFCApplication_IPtestView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMFCApplicationIPtestView

IMPLEMENT_DYNCREATE(CMFCApplicationIPtestView, CView)

BEGIN_MESSAGE_MAP(CMFCApplicationIPtestView, CView)
	// 표준 인쇄 명령입니다.
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CMFCApplicationIPtestView::OnFilePrintPreview)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
	ON_COMMAND(ID_DOWN_SAMPLING, &CMFCApplicationIPtestView::OnDownSampling)
	ON_COMMAND(ID_UP_SAMPLING, &CMFCApplicationIPtestView::OnUpSampling)
	ON_COMMAND(ID_SUM_CONSTANT, &CMFCApplicationIPtestView::OnSumConstant)
	ON_COMMAND(ID_SUB_CONSTANT, &CMFCApplicationIPtestView::OnSubConstant)
	ON_COMMAND(ID_MUL_CONSTANT, &CMFCApplicationIPtestView::OnMulConstant)
	ON_COMMAND(ID_DIV_CONSTANT, &CMFCApplicationIPtestView::OnDivConstant)
	ON_COMMAND(ID_AND_OPERATE, &CMFCApplicationIPtestView::OnAndOperate)
	ON_COMMAND(ID_OR_OPERATE, &CMFCApplicationIPtestView::OnOrOperate)
	ON_COMMAND(ID_XOR_OPERATE, &CMFCApplicationIPtestView::OnXorOperate)
	ON_COMMAND(ID_NEGA_TRANSFORM, &CMFCApplicationIPtestView::OnNegaTransform)
	ON_COMMAND(ID_GAMMA_CORRECTION, &CMFCApplicationIPtestView::OnGammaCorrection)
	ON_COMMAND(ID_BINARIZATION, &CMFCApplicationIPtestView::OnBinarization)
	ON_COMMAND(ID_STRESS_TRANSFORM, &CMFCApplicationIPtestView::OnStressTransform)
END_MESSAGE_MAP()

// CMFCApplicationIPtestView 생성/소멸

CMFCApplicationIPtestView::CMFCApplicationIPtestView() noexcept
{
	// TODO: 여기에 생성 코드를 추가합니다.

}

CMFCApplicationIPtestView::~CMFCApplicationIPtestView()
{
}

BOOL CMFCApplicationIPtestView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CView::PreCreateWindow(cs);
}

// CMFCApplicationIPtestView 그리기

void CMFCApplicationIPtestView::OnDraw(CDC* pDC)
{
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: 여기에 원시 데이터에 대한 그리기 코드를 추가합니다.

	int i, j;
	unsigned char R, G, B;

	for (i = 0; i < pDoc->m_height; i++) {
		for (j = 0; j < pDoc->m_width; j++) {
			R = G = B = pDoc->m_InputImage[i * pDoc->m_width + j];
			pDC->SetPixel(j + 5, i + 5, RGB(R, G, B));
		}
	}

	// 축소된 영상 출력
	for (i = 0; i < pDoc->m_Re_height; i++) {
		for (j = 0; j < pDoc->m_Re_width; j++) {
			R = pDoc->m_OutputImage[i * pDoc->m_Re_width + j];
			G = B = R;
			pDC->SetPixel(j + pDoc->m_width + 10, i + 5, RGB(R, G, B));
		}
	}
}


// CMFCApplicationIPtestView 인쇄


void CMFCApplicationIPtestView::OnFilePrintPreview()
{
#ifndef SHARED_HANDLERS
	AFXPrintPreview(this);
#endif
}

BOOL CMFCApplicationIPtestView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 기본적인 준비
	return DoPreparePrinting(pInfo);
}

void CMFCApplicationIPtestView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄하기 전에 추가 초기화 작업을 추가합니다.
}

void CMFCApplicationIPtestView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄 후 정리 작업을 추가합니다.
}

void CMFCApplicationIPtestView::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CMFCApplicationIPtestView::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// CMFCApplicationIPtestView 진단

#ifdef _DEBUG
void CMFCApplicationIPtestView::AssertValid() const
{
	CView::AssertValid();
}

void CMFCApplicationIPtestView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CMFCApplicationIPtestDoc* CMFCApplicationIPtestView::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCApplicationIPtestDoc)));
	return (CMFCApplicationIPtestDoc*)m_pDocument;
}
#endif //_DEBUG


// CMFCApplicationIPtestView 메시지 처리기

void CMFCApplicationIPtestView::OnDownSampling()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument(); // Doc 클래스 참조
	ASSERT_VALID(pDoc);

	pDoc->OnDownSampling(); // Doc 클래스에 OnDownSampling 함수 호출

	Invalidate(TRUE); // 화면 갱신
}

void CMFCApplicationIPtestView::OnUpSampling()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	CMFCApplicationIPtestDoc* pDoc = GetDocument(); // Doc 클래스 참조
	ASSERT_VALID(pDoc);

	pDoc->OnUpSampling(); // Doc 클래스에 OnUpSampling 함수 호출

	Invalidate(TRUE); // 화면 갱신

}

void CMFCApplicationIPtestView::OnSumConstant()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	// 도큐먼트 클래스 참조
	ASSERT_VALID(pDoc); // 인스턴스 주소를 가져옴

	pDoc->OnSumConstant();

	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnSubConstant()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();// 도큐먼트 클래스 참조
	ASSERT_VALID(pDoc); // 인스턴스 주소를 가져옴

	pDoc->OnSubConstant();

	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnMulConstant()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument(); // 도큐먼트 클래스 참조
	ASSERT_VALID(pDoc); // 인스턴스 주소를 가져옴

	pDoc->OnMulConstant();

	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnDivConstant()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument(); // 도큐먼트 클래스 참조
	ASSERT_VALID(pDoc); // 인스턴스 주소를 가져옴

	pDoc->OnDivConstant();

	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnAndOperate()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);

	pDoc->OnAndOperate();

	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnOrOperate()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);

	pDoc->OnOrOperate();

	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnXorOperate()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnXorOperate();
	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnNegaTransform()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);

	pDoc->OnNegaTransform();

	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnGammaCorrection()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);

	pDoc->OnGammaCorrection();

	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnBinarization()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);

	pDoc->OnBinarization();

	Invalidate(TRUE);
}

void CMFCApplicationIPtestView::OnStressTransform()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);

	pDoc->OnStressTransform();

	Invalidate(TRUE);
}
