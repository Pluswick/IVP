
// MFC_assignment04View.cpp: CMFCassignment04View 클래스의 구현
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "MFC_assignment04.h"
#endif

#include "MFC_assignment04Doc.h"
#include "MFC_assignment04View.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMFCassignment04View

IMPLEMENT_DYNCREATE(CMFCassignment04View, CView)

BEGIN_MESSAGE_MAP(CMFCassignment04View, CView)
	// 표준 인쇄 명령입니다.
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CMFCassignment04View::OnFilePrintPreview)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
	ON_COMMAND(ID_UP_SAMPLING, &CMFCassignment04View::OnUpSampling)
END_MESSAGE_MAP()

// CMFCassignment04View 생성/소멸

CMFCassignment04View::CMFCassignment04View() noexcept
{
	// TODO: 여기에 생성 코드를 추가합니다.

}

CMFCassignment04View::~CMFCassignment04View()
{
}

BOOL CMFCassignment04View::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CView::PreCreateWindow(cs);
}

// CMFCassignment04View 그리기

void CMFCassignment04View::OnDraw(CDC* pDC)
{
	CMFCassignment04Doc* pDoc = GetDocument();
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

	// 처리된 영상 출력
	for (i = 0; i < pDoc->m_Re_height; i++) {
		for (j = 0; j < pDoc->m_Re_width; j++) {
			R = pDoc->m_OutputImage[i * pDoc->m_Re_width + j];
			G = B = R;
			pDC->SetPixel(j + pDoc->m_width + 10, i + 5, RGB(R, G, B));
		}
	}


}


// CMFCassignment04View 인쇄


void CMFCassignment04View::OnFilePrintPreview()
{
#ifndef SHARED_HANDLERS
	AFXPrintPreview(this);
#endif
}

BOOL CMFCassignment04View::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 기본적인 준비
	return DoPreparePrinting(pInfo);
}

void CMFCassignment04View::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄하기 전에 추가 초기화 작업을 추가합니다.
}

void CMFCassignment04View::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄 후 정리 작업을 추가합니다.
}

void CMFCassignment04View::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CMFCassignment04View::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// CMFCassignment04View 진단

#ifdef _DEBUG
void CMFCassignment04View::AssertValid() const
{
	CView::AssertValid();
}

void CMFCassignment04View::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CMFCassignment04Doc* CMFCassignment04View::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCassignment04Doc)));
	return (CMFCassignment04Doc*)m_pDocument;
}
#endif //_DEBUG


// CMFCassignment04View 메시지 처리기

void CMFCassignment04View::OnUpSampling()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CMFCassignment04Doc* pDoc = GetDocument(); // Doc 클래스 참조
	ASSERT_VALID(pDoc);

	pDoc->OnUpSampling(); // Doc 클래스에 OnUpSampling 함수 호출

	Invalidate(TRUE); // 화면 갱신

}
