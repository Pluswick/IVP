
// MFCApplication_IPtestDoc.cpp: CMFCApplicationIPtestDoc 클래스의 구현
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "MFCApplication_IPtest.h"
#endif

#include "MFCApplication_IPtestDoc.h"
#include "CDownSampleDlg.h"
#include "CUpSampleDlg.h"
#include "CConstantDlg.h"
#include "CStressTransformDlg.h" // 범위 강조 대화상자를 위한 헤더 선언
#include <propkey.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMFCApplicationIPtestDoc

IMPLEMENT_DYNCREATE(CMFCApplicationIPtestDoc, CDocument)

BEGIN_MESSAGE_MAP(CMFCApplicationIPtestDoc, CDocument)
	ON_COMMAND(ID_SUM_CONSTANT, &CMFCApplicationIPtestDoc::OnSumConstant)
	ON_COMMAND(ID_SUB_CONSTANT, &CMFCApplicationIPtestDoc::OnSubConstant)
	ON_COMMAND(ID_MUL_CONSTANT, &CMFCApplicationIPtestDoc::OnMulConstant)
	ON_COMMAND(ID_DIV_CONSTANT, &CMFCApplicationIPtestDoc::OnDivConstant)
	ON_COMMAND(ID_AND_OPERATE, &CMFCApplicationIPtestDoc::OnAndOperate)
	ON_COMMAND(ID_OR_OPERATE, &CMFCApplicationIPtestDoc::OnOrOperate)
	ON_COMMAND(ID_XOR_OPERATE, &CMFCApplicationIPtestDoc::OnXorOperate)
	ON_COMMAND(ID_NEGA_TRANSFORM, &CMFCApplicationIPtestDoc::OnNegaTransform)
	ON_COMMAND(ID_GAMMA_CORRECTION, &CMFCApplicationIPtestDoc::OnGammaCorrection)
	ON_COMMAND(ID_BINARIZATION, &CMFCApplicationIPtestDoc::OnBinarization)
	ON_COMMAND(ID_STRESS_TRANSFORM, &CMFCApplicationIPtestDoc::OnStressTransform)
END_MESSAGE_MAP()


// CMFCApplicationIPtestDoc 생성/소멸
CMFCApplicationIPtestDoc::CMFCApplicationIPtestDoc() noexcept
{
	// TODO: 여기에 일회성 생성 코드를 추가합니다.

}

CMFCApplicationIPtestDoc::~CMFCApplicationIPtestDoc()
{
}

BOOL CMFCApplicationIPtestDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: 여기에 재초기화 코드를 추가합니다.
	// SDI 문서는 이 문서를 다시 사용합니다.

	return TRUE;
}




// CMFCApplicationIPtestDoc serialization

void CMFCApplicationIPtestDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: 여기에 저장 코드를 추가합니다.
	}
	else
	{
		// TODO: 여기에 로딩 코드를 추가합니다.
	}
}

#ifdef SHARED_HANDLERS

// 축소판 그림을 지원합니다.
void CMFCApplicationIPtestDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// 문서의 데이터를 그리려면 이 코드를 수정하십시오.
	dc.FillSolidRect(lprcBounds, RGB(255, 255, 255));

	CString strText = _T("TODO: implement thumbnail drawing here");
	LOGFONT lf;

	CFont* pDefaultGUIFont = CFont::FromHandle((HFONT) GetStockObject(DEFAULT_GUI_FONT));
	pDefaultGUIFont->GetLogFont(&lf);
	lf.lfHeight = 36;

	CFont fontDraw;
	fontDraw.CreateFontIndirect(&lf);

	CFont* pOldFont = dc.SelectObject(&fontDraw);
	dc.DrawText(strText, lprcBounds, DT_CENTER | DT_WORDBREAK);
	dc.SelectObject(pOldFont);
}

// 검색 처리기를 지원합니다.
void CMFCApplicationIPtestDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// 문서의 데이터에서 검색 콘텐츠를 설정합니다.
	// 콘텐츠 부분은 ";"로 구분되어야 합니다.

	// 예: strSearchContent = _T("point;rectangle;circle;ole object;");
	SetSearchContent(strSearchContent);
}

void CMFCApplicationIPtestDoc::SetSearchContent(const CString& value)
{
	if (value.IsEmpty())
	{
		RemoveChunk(PKEY_Search_Contents.fmtid, PKEY_Search_Contents.pid);
	}
	else
	{
		CMFCFilterChunkValueImpl *pChunk = nullptr;
		ATLTRY(pChunk = new CMFCFilterChunkValueImpl);
		if (pChunk != nullptr)
		{
			pChunk->SetTextValue(PKEY_Search_Contents, value, CHUNK_TEXT);
			SetChunkValue(pChunk);
		}
	}
}

#endif // SHARED_HANDLERS

// CMFCApplicationIPtestDoc 진단

#ifdef _DEBUG
void CMFCApplicationIPtestDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CMFCApplicationIPtestDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CMFCApplicationIPtestDoc 명령

BOOL CMFCApplicationIPtestDoc::OnOpenDocument(LPCTSTR lpszPathName)
{
	if (!CDocument::OnOpenDocument(lpszPathName))
		return FALSE;

	// TODO:  여기에 특수화된 작성 코드를 추가합니다.

	CFile File; // 파일 객체 선언

	File.Open(lpszPathName, CFile::modeRead | CFile::typeBinary);
	// 파일 열기 대화상자에서 선택한 파일을 지정하고 읽기 모드 선택

	// 이 책에서는 영상의 크기 256*256, 512*512, 640*480만을 사용한다.
	if (File.GetLength() == 256 * 256) { // RAW 파일의 크기 결정

		m_height = 256;
		m_width = 256;
	}
	else if (File.GetLength() == 512 * 512) { // RAW 파일의 크기 결정
		m_height = 512;
		m_width = 512;
	}
	else if (File.GetLength() == 640 * 480) { // RAW 파일의 크기 결정
		m_height = 480;
		m_width = 640;
	}
	else {
		AfxMessageBox(L"Not Support Image Size"); // 해당 크기가 없는 경우
		return 0;
	}
	m_size = m_width * m_height; // 영상의 크기 계산

	m_InputImage = new unsigned char[m_size];
	// 입력 영상의 크기에 맞는 메모리 할당

	for (int i = 0; i < m_size; i++)
		m_InputImage[i] = 255; // 초기화
	File.Read(m_InputImage, m_size); // 입력 영상 파일 읽기
	File.Close(); // 파일 닫기


	return TRUE;
}

void CMFCApplicationIPtestDoc::OnDownSampling()
{
	// TODO: 여기에 구현 코드 추가.
	int i, j;
	CDownSampleDlg dlg;
	if (dlg.DoModal() == IDOK) // 대화상자의 활성화 여부
	{
		m_Re_height = m_height / dlg.m_DownSampleRate;
		// 축소 영상의 세로 길이를 계산
		m_Re_width = m_width / dlg.m_DownSampleRate;
		// 축소 영상의 가로 길이를 계산
		m_Re_size = m_Re_height * m_Re_width;
		// 축소 영상의 크기를 계산

		m_OutputImage = new unsigned char[m_Re_size];
		// 축소 영상을 위한 메모리 할당

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j]
					= m_InputImage[(i * dlg.m_DownSampleRate * m_width) + dlg.m_DownSampleRate * j];
				// 축소 영상을 생성
			}
		}
	}
}

void CMFCApplicationIPtestDoc::OnUpSampling()
{
	// TODO: 여기에 구현 코드 추가.

	int i, j;

	CUpSampleDlg dlg;
	if (dlg.DoModal() == IDOK) { // DoModal 대화상자의 활성화 여부
		m_Re_height = m_height * dlg.m_UpSampleRate;
		// 확대 영상의 세로 길이 계산
		m_Re_width = m_width * dlg.m_UpSampleRate;
		// 확대 영상의 가로 길이 계산
		m_Re_size = m_Re_height * m_Re_width;
		// 확대 영상의 크기 계산
		m_OutputImage = new unsigned char[m_Re_size];
		// 확대 영상을 위한 메모리 할당

		for (i = 0; i < m_Re_size; i++)
			m_OutputImage[i] = 0; // 초기화

		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				m_OutputImage[i * dlg.m_UpSampleRate * m_Re_width +
					dlg.m_UpSampleRate * j] = m_InputImage[i * m_width + j];
			} // 재배치하여 영상 확대
		}
	}
}

void CMFCApplicationIPtestDoc::OnSumConstant()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	CConstantDlg dlg; // 상수 값을 입력받는 대화상자

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] + dlg.m_Constant >= 255)
				m_OutputImage[i] = 255;
			// 출력 값이 255보다 크면 255 출력
			else
				m_OutputImage[i] = (unsigned char)(m_InputImage[i] + dlg.m_Constant);
			// 상수 값과 화소 값과의 덧셈
		}
	}
}

void CMFCApplicationIPtestDoc::OnSubConstant()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] - dlg.m_Constant < 0)
				m_OutputImage[i] = 0; // 출력 값이 255보다 크면 255를 출력
			else
				m_OutputImage[i] = (unsigned char)(m_InputImage[i] - dlg.m_Constant);
			// 상수 값과 화소 값과의 뺄셈
		}
	}
}

void CMFCApplicationIPtestDoc::OnMulConstant()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] * dlg.m_Constant > 255)
				m_OutputImage[i] = 255;
			// 곱의 값이 255보다 크면 255를 출력
			else if (m_InputImage[i] * dlg.m_Constant < 0)
				m_OutputImage[i] = 0;
			// 곱의 값이 0보다 작으면 0을 출력
			else
				m_OutputImage[i]
				= (unsigned char)(m_InputImage[i] * dlg.m_Constant);
				// 상수 값과 화소 값 곱셈
		}
	}
}

void CMFCApplicationIPtestDoc::OnDivConstant()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] / dlg.m_Constant > 255)
				m_OutputImage[i] = 255;
			// 나눗셈의 값이 255보다 크면 255를 출력
			else if (m_InputImage[i] / dlg.m_Constant < 0)
				m_OutputImage[i] = 0;
			// 나눗셈의 값이 0보다 작으면 0을 출력
			else
				m_OutputImage[i]
				= (unsigned char)(m_InputImage[i] / dlg.m_Constant);
				// 상수 값과 화소 값 나눗셈
		}
	}
}

void CMFCApplicationIPtestDoc::OnAndOperate()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CConstantDlg dlg;
	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			// 비트 단위 AND 연산
			if ((m_InputImage[i] & (unsigned char)dlg.m_Constant) >= 255)
			{
				m_OutputImage[i] = 255;
			}
			else if ((m_InputImage[i] & (unsigned char)dlg.m_Constant) < 0)
			{
				m_OutputImage[i] = 0;
			}
			else {
				m_OutputImage[i] = (m_InputImage[i]
					& (unsigned char)dlg.m_Constant);
			}
		}
	}
}

void CMFCApplicationIPtestDoc::OnOrOperate()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CConstantDlg dlg;
	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			// 비트 단위 OR 연산
			if ((m_InputImage[i] | (unsigned char)dlg.m_Constant) >= 255) {
				m_OutputImage[i] = 255;
			}
			else if ((m_InputImage[i] | (unsigned char)dlg.m_Constant) < 0) {
				m_OutputImage[i] = 0;
			}
			else {
				m_OutputImage[i] = (m_InputImage[i] |
					(unsigned char)dlg.m_Constant);
			}
		}
	}
}

void CMFCApplicationIPtestDoc::OnXorOperate()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CConstantDlg dlg;
	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			// 비트 단위 XOR 연산
			if ((m_InputImage[i] ^ (unsigned char)dlg.m_Constant) >= 255) {
				m_OutputImage[i] = 255;
			}
			else if ((m_InputImage[i] ^ (unsigned char)dlg.m_Constant) < 0) {
				m_OutputImage[i] = 0;
			}
			else {
				m_OutputImage[i] = (m_InputImage[i]
					^ (unsigned char)dlg.m_Constant);
			}
		}
	}
}
void CMFCApplicationIPtestDoc::OnNegaTransform()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	for (i = 0; i < m_size; i++)
		m_OutputImage[i] = 255 - m_InputImage[i]; // 영상 반전을 수행
}

void CMFCApplicationIPtestDoc::OnGammaCorrection()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CConstantDlg dlg;

	int i;
	double temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			temp = pow(m_InputImage[i], 1 / dlg.m_Constant);
			// 감마 값 계산
			if (temp < 0)
				m_OutputImage[i] = 0;
			else if (temp > 255)
				m_OutputImage[i] = 255;
			else
				m_OutputImage[i] = (unsigned char)temp;
		}
	}
}

void CMFCApplicationIPtestDoc::OnBinarization()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] >= dlg.m_Constant)
				m_OutputImage[i] = 255; // 임계 값보다 크면 255 출력
			else
				m_OutputImage[i] = 0; // 임계 값보다 작으면 0 출력
		}
	}
}

void CMFCApplicationIPtestDoc::OnStressTransform()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CStressTransformDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			// 입력 값이 강조 시작 값과 강조 종료 값 사이에 위치하면 255 출력
			if (m_InputImage[i] >= dlg.m_StartPoint &&
				m_InputImage[i] <= dlg.m_EndPoint)
				m_OutputImage[i] = 255;
			else
				m_OutputImage[i] = m_InputImage[i];
		}
	}
}
