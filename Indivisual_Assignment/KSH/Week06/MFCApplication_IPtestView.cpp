
// MFCApplication_IPtestView.cpp: CMFCApplicationIPtestView Ŭ������ ����
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS�� �̸� ����, ����� �׸� �� �˻� ���� ó���⸦ �����ϴ� ATL ������Ʈ���� ������ �� ������
// �ش� ������Ʈ�� ���� �ڵ带 �����ϵ��� �� �ݴϴ�.
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
	// ǥ�� �μ� ����Դϴ�.
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CMFCApplicationIPtestView::OnFilePrintPreview)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
	ON_COMMAND(ID_DOWN_SAMPLING, &CMFCApplicationIPtestView::OnDownSampling)
	ON_COMMAND(ID_UP_SAMPLING, &CMFCApplicationIPtestView::OnUpSampling)
	ON_COMMAND(ID_QUANTIZATION, &CMFCApplicationIPtestView::OnQuantization)
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
	ON_COMMAND(ID_END_IN_SEARCH, &CMFCApplicationIPtestView::OnEndInSearch)
	ON_COMMAND(ID_HISTOGRAM, &CMFCApplicationIPtestView::OnHistogram)
	ON_COMMAND(ID_HISTO_EQUAL, &CMFCApplicationIPtestView::OnHistoEqual)
	ON_COMMAND(ID_HISTO_SPEC, &CMFCApplicationIPtestView::OnHistoSpec)
END_MESSAGE_MAP()

// CMFCApplicationIPtestView ����/�Ҹ�

CMFCApplicationIPtestView::CMFCApplicationIPtestView() noexcept
{
	// TODO: ���⿡ ���� �ڵ带 �߰��մϴ�.

}

CMFCApplicationIPtestView::~CMFCApplicationIPtestView()
{
}

BOOL CMFCApplicationIPtestView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs�� �����Ͽ� ���⿡��
	//  Window Ŭ���� �Ǵ� ��Ÿ���� �����մϴ�.

	return CView::PreCreateWindow(cs);
}

// CMFCApplicationIPtestView �׸���

void CMFCApplicationIPtestView::OnDraw(CDC* pDC)
{
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	int i, j;
	unsigned char R, G, B;
	// �Է� ���� ���
	for (i = 0; i < pDoc->m_height; i++) {
		for (j = 0; j < pDoc->m_width; j++) {
			R = pDoc->m_InputImage[i * pDoc->m_width + j];
			G = B = R;
			pDC->SetPixel(j + 5, i + 5, RGB(R, G, B));
		}
	}
	// ��ҵ� ���� ���
	for (i = 0; i < pDoc->m_Re_height; i++) {
		for (j = 0; j < pDoc->m_Re_width; j++) {
			R = pDoc->m_OutputImage[i * pDoc->m_Re_width + j];
			G = B = R;
			pDC->SetPixel(j + pDoc->m_width + 10, i + 5, RGB(R, G, B));
		}
	}
}


// CMFCApplicationIPtestView �μ�


void CMFCApplicationIPtestView::OnFilePrintPreview()
{
#ifndef SHARED_HANDLERS
	AFXPrintPreview(this);
#endif
}

BOOL CMFCApplicationIPtestView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// �⺻���� �غ�
	return DoPreparePrinting(pInfo);
}

void CMFCApplicationIPtestView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: �μ��ϱ� ���� �߰� �ʱ�ȭ �۾��� �߰��մϴ�.
}

void CMFCApplicationIPtestView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: �μ� �� ���� �۾��� �߰��մϴ�.
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


// CMFCApplicationIPtestView ����

#ifdef _DEBUG
void CMFCApplicationIPtestView::AssertValid() const
{
	CView::AssertValid();
}

void CMFCApplicationIPtestView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CMFCApplicationIPtestDoc* CMFCApplicationIPtestView::GetDocument() const // ����׵��� ���� ������ �ζ������� �����˴ϴ�.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCApplicationIPtestDoc)));
	return (CMFCApplicationIPtestDoc*)m_pDocument;
}
#endif //_DEBUG


// CMFCApplicationIPtestView �޽��� ó����


void CMFCApplicationIPtestView::OnDownSampling()
{
	CMFCApplicationIPtestDoc* pDoc = GetDocument(); // Doc Ŭ���� ����
	ASSERT_VALID(pDoc);
	pDoc->OnDownSampling(); // Doc Ŭ������ OnDownSampling �Լ� ȣ��
	Invalidate(TRUE); // ȭ�� ����
}


void CMFCApplicationIPtestView::OnUpSampling()
{

	// TODO: Add your command handler code here
	CMFCApplicationIPtestDoc* pDoc = GetDocument(); // Doc Ŭ���� ����
	ASSERT_VALID(pDoc);

	pDoc->OnUpSampling(); // Doc Ŭ������ OnUpSampling �Լ� ȣ��

	Invalidate(TRUE); // ȭ�� ����
}



void CMFCApplicationIPtestView::OnQuantization()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument(); // Doc Ŭ���� ����
	ASSERT_VALID(pDoc);

	pDoc->OnQuantization(); // Doc Ŭ������ OnQuantization �Լ� ȣ��

	Invalidate(TRUE); // ȭ�� ����
}


void CMFCApplicationIPtestView::OnSumConstant()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.

	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	// ��ť��Ʈ Ŭ���� ����
	ASSERT_VALID(pDoc); // �ν��Ͻ� �ּҸ� ������
	pDoc->OnSumConstant();
	Invalidate(TRUE);
}



void CMFCApplicationIPtestView::OnSubConstant()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();// ��ť��Ʈ Ŭ���� ����
	ASSERT_VALID(pDoc); // �ν��Ͻ� �ּҸ� ������
	pDoc->OnSubConstant();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnMulConstant()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument(); // ��ť��Ʈ Ŭ���� ����
	ASSERT_VALID(pDoc); // �ν��Ͻ� �ּҸ� ������
	pDoc->OnMulConstant();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnDivConstant()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument(); // ��ť��Ʈ Ŭ���� ����
	ASSERT_VALID(pDoc); // �ν��Ͻ� �ּҸ� ������
	pDoc->OnDivConstant();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnAndOperate()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnAndOperate();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnOrOperate()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnOrOperate();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnXorOperate()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnXorOperate();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnNegaTransform()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnNegaTransform();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnGammaCorrection()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnGammaCorrection();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnBinarization()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnBinarization();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnStressTransform()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnStressTransform();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnEndInSearch()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnEndInSearch();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnHistogram()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnHistogram();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnHistoEqual()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);

	pDoc->OnHistoEqual();
	Invalidate(TRUE);
}


void CMFCApplicationIPtestView::OnHistoSpec()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CMFCApplicationIPtestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	pDoc->OnHistoSpec();
	Invalidate(TRUE);
}
