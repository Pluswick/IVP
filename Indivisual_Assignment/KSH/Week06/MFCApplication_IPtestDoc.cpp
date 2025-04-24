
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

#include "CDownSampleDlg.h" // 대화상자 사용을 위한 헤더 선언
#include "CUpSampleDlg.h"
#include "CQuantizationDlg.h"
#include "math.h" // 수학 함수 사용을 위한 헤더 선언
#include "CConstantDlg.h"
#include "CStressTransformDlg.h"
#include <propkey.h>
//#include "stdafx.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMFCApplicationIPtestDoc

IMPLEMENT_DYNCREATE(CMFCApplicationIPtestDoc, CDocument)

BEGIN_MESSAGE_MAP(CMFCApplicationIPtestDoc, CDocument)
	ON_COMMAND(ID_DOWN_SAMPLING, &CMFCApplicationIPtestDoc::OnDownSampling)
	ON_COMMAND(ID_QUANTIZATION, &CMFCApplicationIPtestDoc::OnQuantization)
//	ON_COMMAND(ID_FILE_SAVE, &CMFCApplicationIPtestDoc::OnSaveDocument)
	ON_COMMAND(ID_FILE_SAVE, &CMFCApplicationIPtestDoc::OnFileSave)
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
	ON_COMMAND(ID_END_IN_SEARCH, &CMFCApplicationIPtestDoc::OnEndInSearch)
	ON_COMMAND(ID_HISTOGRAM, &CMFCApplicationIPtestDoc::OnHistogram)
	ON_COMMAND(ID_HISTO_EQUAL, &CMFCApplicationIPtestDoc::OnHistoEqual)
	ON_COMMAND(ID_HISTO_SPEC, &CMFCApplicationIPtestDoc::OnHistoSpec)
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
		m_InputImage[i] = 255; // 초기화 -> 모든 픽셀에 255값
	File.Read(m_InputImage, m_size); // 입력 영상 파일 읽기
	File.Close(); // 파일 닫기
	return TRUE;
}


void CMFCApplicationIPtestDoc::OnDownSampling()
{
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





void CMFCApplicationIPtestDoc::OnQuantization()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CQuantizationDlg dlg;
	if (dlg.DoModal() == IDOK)
		// 양자화 비트 수를 결정하는 대화상자의 활성화 여부
	{
		int i, j, value, LEVEL;
		double HIGH, * TEMP;
		m_Re_height = m_height;		//크기 조정은 없다
		m_Re_width = m_width;
		m_Re_size = m_Re_height * m_Re_width;
		m_OutputImage = new unsigned char[m_Re_size];
		// 양자화 처리된 영상을 출력하기 위한 메모리 할당
		TEMP = new double[m_size];
		// 입력 영상 크기(m_size)와 동일한 메모리 할당
		LEVEL = 256; // 입력 영상의 양자화 단계(28=256)
		HIGH = 256.;
		value = (int)pow(2, dlg.m_QuantBit);	//pow(2, a) 는 2^a
												//dialog에서 받는 값이 1에서 8사이인 이유 -> 범위가 0 ~ 256
		// 양자화 단계 결정(예 : 24=16)
		for (i = 0; i < m_size; i++) {
			for (j = 0; j < value; j++) {
				if (m_InputImage[i] >= (LEVEL / value) * j &&
					m_InputImage[i] < (LEVEL / value) * (j + 1)) {
					TEMP[i] = (double)(HIGH / value) * j; // 양자화 수행
				}
			}
		}
		for (i = 0; i < m_size; i++) {
			m_OutputImage[i] = (unsigned char)TEMP[i];
			// 결과 영상 생성
		}
	}
}




BOOL CMFCApplicationIPtestDoc::OnSaveDocument(LPCTSTR lpszPathName)
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
	CFile File; // 파일 객체 선언
	CFileDialog SaveDlg(FALSE, _T("raw"), NULL, OFN_HIDEREADONLY);
	// raw 파일을 다른 이름으로 저장하기를 위한 대화상자 객체 선언
	if (SaveDlg.DoModal() == IDOK) {
		// DoModal 멤버 함수에서 저장하기 수행
		File.Open(SaveDlg.GetPathName(), CFile::modeCreate |
			CFile::modeWrite);
		// 파일 열기
		File.Write(m_InputImage, m_size); // 파일 쓰기
		File.Close();
		// 파일 닫기
	}
	return TRUE;
}

void CMFCApplicationIPtestDoc::OnFileSave()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	OnSaveDocument(GetPathName());
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
				m_OutputImage[i]
				= (unsigned char)(m_InputImage[i] - dlg.m_Constant);
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
				m_OutputImage[i] = (m_InputImage[i] | (unsigned char)dlg.m_Constant);
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
				m_OutputImage[i] = (m_InputImage[i]	^ (unsigned char)dlg.m_Constant);
			}
		}
	}
}


void CMFCApplicationIPtestDoc::OnNegaTransform()
{
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


void CMFCApplicationIPtestDoc::OnEndInSearch()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	int i;
	unsigned char LOW, HIGH, MAX, MIN;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	LOW = 0;
	HIGH = 255;
	MIN = m_InputImage[0];
	MAX = m_InputImage[0];
	for (i = 0; i < m_size; i++) {
		if (m_InputImage[i] < MIN)
			MIN = m_InputImage[i];
	}
	for (i = 0; i < m_size; i++) {
		if (m_InputImage[i] > MAX)
			MAX = m_InputImage[i];
	}
	m_OutputImage = new unsigned char[m_Re_size];
	for (i = 0; i < m_size; i++)
		{
			// 원본 영상의 최소값보다 작은 값은 0
			if (m_InputImage[i] <= MIN)
			{
				m_OutputImage[i] = 0;
			}
			// 원본 영상의 최대값보다 큰 값은 255
			else if (m_InputImage[i] >= MAX)
			{
				m_OutputImage[i] = 255;
			}
			else
			m_OutputImage[i] = (unsigned char)((m_InputImage[i] - MIN) * HIGH / (MAX - MIN));
	}

}

double m_HIST[256];
double m_Sum_Of_HIST[256];
unsigned char m_Scale_HIST[256];


void CMFCApplicationIPtestDoc::OnHistogram()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	// 히스토그램의 값은 0~255
	// 히스토그램의 크기 값을 MAX=255로 정규화하여 출력
	// 히스트그램의 크기 : 256*256 지정
	int i, j, value;
	unsigned char LOW, HIGH;
	double MAX, MIN, DIF;
	m_Re_height = 256;
	m_Re_width = 256;
	m_Re_size = m_Re_height * m_Re_width;
	LOW = 0;
	HIGH = 255;
	// 초기화
	for (i = 0; i < 256; i++)
		m_HIST[i] = LOW;
	// 빈도 수 조사
	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
	}
	// 정규화
	MAX = m_HIST[0];
	MIN = m_HIST[0];
	for (i = 0; i < 256; i++) {
		if (m_HIST[i] > MAX)
			MAX = m_HIST[i];
	}
	for (i = 0; i < 256; i++) {
		if (m_HIST[i] < MIN)
			MIN = m_HIST[i];
	}
	
	DIF = MAX - MIN;
	// 정규화된 히스토그램
	for (i = 0; i < 256; i++)
		m_Scale_HIST[i] = (unsigned char)((m_HIST[i] - MIN) * HIGH / DIF);
	// 정규화된 히스토그램 출력
	m_OutputImage = new unsigned char[m_Re_size + (256 * 20)];
	for (i = 0; i < m_Re_size; i++)
		m_OutputImage[i] = 255;
	// 정규화된 히스토그램의 값은 출력 배열에 검은 점(0)으로 표현
	for (i = 0; i < 256; i++) {
		for (j = 0; j < m_Scale_HIST[i]; j++) {
			m_OutputImage[m_Re_width * (m_Re_height - j - 1) + i] = 0;
		}
	}
	// 히스토그램을 출력하고 그 아래 부분에 히스토그램의 색을 표시
	for (i = m_Re_height; i < m_Re_height + 5; i++) {
		for (j = 0; j < 256; j++) {
			m_OutputImage[m_Re_height * i + j] = 255;
		}
	}
	for (i = m_Re_height + 5; i < m_Re_height + 20; i++) {
		for (j = 0; j < 256; j++) {
			m_OutputImage[m_Re_height * i + j] = j;
		}
	}
	m_Re_height = m_Re_height + 20;
	m_Re_size = m_Re_height * m_Re_width;
}


void CMFCApplicationIPtestDoc::OnHistoEqual()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	int i, value;
	unsigned char LOW, HIGH, Temp;
	double SUM = 0.0;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	LOW = 0;
	HIGH = 255;
	// 초기화
	for (i = 0; i < 256; i++)
		m_HIST[i] = LOW;
	// 빈도 수 조사
	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
	}
	// 누적 히스토그램 생성
	for (i = 0; i < 256; i++) {
		SUM += m_HIST[i];
		m_Sum_Of_HIST[i] = SUM;
	}
	m_OutputImage = new unsigned char[m_Re_size];
	// 입력 영상을 평활화된 영상으로 출력
	for (i = 0; i < m_size; i++) {
		Temp = m_InputImage[i];
		m_OutputImage[i] = (unsigned char)(m_Sum_Of_HIST[Temp] * HIGH / m_size);
	}
}


void CMFCApplicationIPtestDoc::OnHistoSpec()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	int i, value, Dvalue, top, bottom, DADD;
	unsigned char* m_DTEMP = nullptr, m_Sum_Of_ScHIST[256], m_TABLE[256];
	unsigned char LOW, HIGH, Temp, * m_Org_Temp;
	double m_DHIST[256], m_Sum_Of_DHIST[256], SUM = 0.0, DSUM = 0.0;
	double DMAX, DMIN;

	top = 255;
	bottom = top - 1;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	m_OutputImage = new unsigned char[m_Re_size];
	m_Org_Temp = new unsigned char[m_size];

	CFile File;
	CFileDialog OpenDlg(TRUE);

	// 원하는 히스토그램이 있는 영상을 입력받음
	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);
		if (File.GetLength() == (unsigned)m_size) {
			m_DTEMP = new unsigned char[m_size];
			File.Read(m_DTEMP, m_size);
			File.Close();
		}
		else {
			AfxMessageBox(L"Image size not matched");
			// 같은 크기의 영상을 대상으로 함
			return;
		}
	}
	LOW = 0;
	HIGH = 255;
	// 초기화
	for (i = 0; i < 256; i++) {
		m_HIST[i] = LOW;
		m_DHIST[i] = LOW;
		m_TABLE[i] = LOW;
	}
	// 빈도 수 조사
	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
		Dvalue = (int)m_DTEMP[i];
		m_DHIST[Dvalue]++;
	}
	// 누적 히스토그램 조사
	for (i = 0; i < 256; i++) {
		SUM += m_HIST[i];
		m_Sum_Of_HIST[i] = SUM;
		DSUM += m_DHIST[i];
		m_Sum_Of_DHIST[i] = DSUM;
	}
	// 원본 영상의 평활화
	for (i = 0; i < m_size; i++) {
		Temp = m_InputImage[i];
		m_Org_Temp[i] = (unsigned char)(m_Sum_Of_HIST[Temp] * HIGH / m_size);
	}
	// 누적 히스토그램에서 최소값과 최대값 지정
	DMIN = m_Sum_Of_DHIST[0];
	DMAX = m_Sum_Of_DHIST[255];
	// 원하는 영상을 평활화
	for (i = 0; i < 256; i++) {
		m_Sum_Of_ScHIST[i] = (unsigned char)((m_Sum_Of_DHIST[i]
			- DMIN) * HIGH / (DMAX - DMIN));
	}
	// 룩업테이블을 이용한 명세화
	for (; ; ) {
		for (i = m_Sum_Of_ScHIST[bottom];
			i <= m_Sum_Of_ScHIST[top]; i++) {
			m_TABLE[i] = top;
		}
		top = bottom;
		bottom = bottom - 1;
		if (bottom < -1)
			break;
	}
	for (i = 0; i < m_size; i++) {
		DADD = (int)m_Org_Temp[i];
		m_OutputImage[i] = m_TABLE[DADD];
	}
	
}

/*
void CMFCApplicationIPtestDoc::OnHistoSpec()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	int i, value, Dvalue, top, bottom, DADD;
	unsigned char* m_DTEMP = nullptr;  // 초기화
	unsigned char m_Sum_Of_ScHIST[256], m_TABLE[256];
	unsigned char LOW, HIGH, Temp, * m_Org_Temp;
	double m_DHIST[256], m_Sum_Of_DHIST[256], SUM = 0.0, DSUM = 0.0;
	double DMAX, DMIN;

	top = 255;
	bottom = top - 1;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	m_OutputImage = new unsigned char[m_Re_size];
	m_Org_Temp = new unsigned char[m_size];

	CFile File;
	CFileDialog OpenDlg(TRUE);

	// 원하는 히스토그램이 있는 영상을 입력받음
	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetFileName(), CFile::modeRead);
		if (File.GetLength() == (unsigned)m_size) {
			m_DTEMP = new unsigned char[m_size];  // 메모리 할당
			File.Read(m_DTEMP, m_size);
			File.Close();
		}
		else {
			AfxMessageBox(L"Image size not matched");
			delete[] m_DTEMP;  // 메모리 해제
			return;
		}
	}
	else {
		AfxMessageBox(L"File selection cancelled");
		delete[] m_DTEMP;  // 메모리 해제
		return;
	}

	LOW = 0;
	HIGH = 255;
	// 초기화
	for (i = 0; i < 256; i++) {
		m_HIST[i] = LOW;
		m_DHIST[i] = LOW;
		m_TABLE[i] = LOW;
	}
	// 빈도 수 조사
	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
		Dvalue = (int)m_DTEMP[i];
		m_DHIST[Dvalue]++;
	}
	// 누적 히스토그램 조사
	for (i = 0; i < 256; i++) {
		SUM += m_HIST[i];
		m_Sum_Of_HIST[i] = SUM;
		DSUM += m_DHIST[i];
		m_Sum_Of_DHIST[i] = DSUM;
	}
	// 원본 영상의 평활화
	for (i = 0; i < m_size; i++) {
		Temp = m_InputImage[i];
		m_Org_Temp[i] = (unsigned char)(m_Sum_Of_HIST[Temp] * HIGH / m_size);
	}
	// 누적 히스토그램에서 최소값과 최대값 지정
	DMIN = m_Sum_Of_DHIST[0];
	DMAX = m_Sum_Of_DHIST[255];
	// 원하는 영상을 평활화
	for (i = 0; i < 256; i++) {
		m_Sum_Of_ScHIST[i] = (unsigned char)((m_Sum_Of_DHIST[i] - DMIN) * HIGH / (DMAX - DMIN));
	}
	// 룩업테이블을 이용한 명세화
	for (; ; ) {
		for (i = m_Sum_Of_ScHIST[bottom]; i <= m_Sum_Of_ScHIST[top]; i++) {
			m_TABLE[i] = top;
		}
		top = bottom;
		bottom = bottom - 1;
		if (bottom < -1)
			break;
	}
	for (i = 0; i < m_size; i++) {
		DADD = (int)m_Org_Temp[i];
		m_OutputImage[i] = m_TABLE[DADD];
	}

	// 사용한 메모리 해제
	delete[] m_DTEMP;
	delete[] m_Org_Temp;
}
*/
