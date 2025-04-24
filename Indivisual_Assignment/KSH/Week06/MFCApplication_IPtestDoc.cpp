
// MFCApplication_IPtestDoc.cpp: CMFCApplicationIPtestDoc Ŭ������ ����
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS�� �̸� ����, ����� �׸� �� �˻� ���� ó���⸦ �����ϴ� ATL ������Ʈ���� ������ �� ������
// �ش� ������Ʈ�� ���� �ڵ带 �����ϵ��� �� �ݴϴ�.
#ifndef SHARED_HANDLERS
#include "MFCApplication_IPtest.h"
#endif

#include "MFCApplication_IPtestDoc.h"

#include "CDownSampleDlg.h" // ��ȭ���� ����� ���� ��� ����
#include "CUpSampleDlg.h"
#include "CQuantizationDlg.h"
#include "math.h" // ���� �Լ� ����� ���� ��� ����
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


// CMFCApplicationIPtestDoc ����/�Ҹ�

CMFCApplicationIPtestDoc::CMFCApplicationIPtestDoc() noexcept
{
	// TODO: ���⿡ ��ȸ�� ���� �ڵ带 �߰��մϴ�.

}

CMFCApplicationIPtestDoc::~CMFCApplicationIPtestDoc()
{
}

BOOL CMFCApplicationIPtestDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: ���⿡ ���ʱ�ȭ �ڵ带 �߰��մϴ�.
	// SDI ������ �� ������ �ٽ� ����մϴ�.

	return TRUE;
}




// CMFCApplicationIPtestDoc serialization

void CMFCApplicationIPtestDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: ���⿡ ���� �ڵ带 �߰��մϴ�.
	}
	else
	{
		// TODO: ���⿡ �ε� �ڵ带 �߰��մϴ�.
	}
}

#ifdef SHARED_HANDLERS

// ����� �׸��� �����մϴ�.
void CMFCApplicationIPtestDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// ������ �����͸� �׸����� �� �ڵ带 �����Ͻʽÿ�.
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

// �˻� ó���⸦ �����մϴ�.
void CMFCApplicationIPtestDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// ������ �����Ϳ��� �˻� �������� �����մϴ�.
	// ������ �κ��� ";"�� ���еǾ�� �մϴ�.

	// ��: strSearchContent = _T("point;rectangle;circle;ole object;");
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

// CMFCApplicationIPtestDoc ����

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


// CMFCApplicationIPtestDoc ���


BOOL CMFCApplicationIPtestDoc::OnOpenDocument(LPCTSTR lpszPathName)
{
	if (!CDocument::OnOpenDocument(lpszPathName))
		return FALSE;

	CFile File; // ���� ��ü ����

	File.Open(lpszPathName, CFile::modeRead | CFile::typeBinary);

	// ���� ���� ��ȭ���ڿ��� ������ ������ �����ϰ� �б� ��� ����
	// �� å������ ������ ũ�� 256*256, 512*512, 640*480���� ����Ѵ�.

	if (File.GetLength() == 256 * 256) { // RAW ������ ũ�� ����
		m_height = 256;
		m_width = 256;
	}
	else if (File.GetLength() == 512 * 512) { // RAW ������ ũ�� ����
		m_height = 512;
		m_width = 512;
	}
	else if (File.GetLength() == 640 * 480) { // RAW ������ ũ�� ����
		m_height = 480;
		m_width = 640;
	}
	else {
		AfxMessageBox(L"Not Support Image Size"); // �ش� ũ�Ⱑ ���� ���
		return 0;
	}
	m_size = m_width * m_height; // ������ ũ�� ���
	m_InputImage = new unsigned char[m_size];
	// �Է� ������ ũ�⿡ �´� �޸� �Ҵ�
	for (int i = 0; i < m_size; i++)
		m_InputImage[i] = 255; // �ʱ�ȭ -> ��� �ȼ��� 255��
	File.Read(m_InputImage, m_size); // �Է� ���� ���� �б�
	File.Close(); // ���� �ݱ�
	return TRUE;
}


void CMFCApplicationIPtestDoc::OnDownSampling()
{
	int i, j;
	CDownSampleDlg dlg;
	if (dlg.DoModal() == IDOK) // ��ȭ������ Ȱ��ȭ ����
	{
		m_Re_height = m_height / dlg.m_DownSampleRate;
		// ��� ������ ���� ���̸� ���
		m_Re_width = m_width / dlg.m_DownSampleRate;
		// ��� ������ ���� ���̸� ���
		m_Re_size = m_Re_height * m_Re_width;
		// ��� ������ ũ�⸦ ���
		m_OutputImage = new unsigned char[m_Re_size];
		// ��� ������ ���� �޸� �Ҵ�
		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j]
					= m_InputImage[(i * dlg.m_DownSampleRate * m_width) + dlg.m_DownSampleRate * j];
				// ��� ������ ����
			}
		}
	}
}


void CMFCApplicationIPtestDoc::OnUpSampling()
{
	int i, j;
	CUpSampleDlg dlg;
	if (dlg.DoModal() == IDOK) { // DoModal ��ȭ������ Ȱ��ȭ ����
		m_Re_height = m_height * dlg.m_UpSampleRate;
		// Ȯ�� ������ ���� ���� ���
		m_Re_width = m_width * dlg.m_UpSampleRate;
		// Ȯ�� ������ ���� ���� ���
		m_Re_size = m_Re_height * m_Re_width;
		// Ȯ�� ������ ũ�� ���
		m_OutputImage = new unsigned char[m_Re_size];
		// Ȯ�� ������ ���� �޸� �Ҵ�
		for (i = 0; i < m_Re_size; i++)
			m_OutputImage[i] = 0; // �ʱ�ȭ
		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				m_OutputImage[i * dlg.m_UpSampleRate * m_Re_width +
					dlg.m_UpSampleRate * j] = m_InputImage[i * m_width + j];
			} // ���ġ�Ͽ� ���� Ȯ��
		}
	}
}





void CMFCApplicationIPtestDoc::OnQuantization()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CQuantizationDlg dlg;
	if (dlg.DoModal() == IDOK)
		// ����ȭ ��Ʈ ���� �����ϴ� ��ȭ������ Ȱ��ȭ ����
	{
		int i, j, value, LEVEL;
		double HIGH, * TEMP;
		m_Re_height = m_height;		//ũ�� ������ ����
		m_Re_width = m_width;
		m_Re_size = m_Re_height * m_Re_width;
		m_OutputImage = new unsigned char[m_Re_size];
		// ����ȭ ó���� ������ ����ϱ� ���� �޸� �Ҵ�
		TEMP = new double[m_size];
		// �Է� ���� ũ��(m_size)�� ������ �޸� �Ҵ�
		LEVEL = 256; // �Է� ������ ����ȭ �ܰ�(28=256)
		HIGH = 256.;
		value = (int)pow(2, dlg.m_QuantBit);	//pow(2, a) �� 2^a
												//dialog���� �޴� ���� 1���� 8������ ���� -> ������ 0 ~ 256
		// ����ȭ �ܰ� ����(�� : 24=16)
		for (i = 0; i < m_size; i++) {
			for (j = 0; j < value; j++) {
				if (m_InputImage[i] >= (LEVEL / value) * j &&
					m_InputImage[i] < (LEVEL / value) * (j + 1)) {
					TEMP[i] = (double)(HIGH / value) * j; // ����ȭ ����
				}
			}
		}
		for (i = 0; i < m_size; i++) {
			m_OutputImage[i] = (unsigned char)TEMP[i];
			// ��� ���� ����
		}
	}
}




BOOL CMFCApplicationIPtestDoc::OnSaveDocument(LPCTSTR lpszPathName)
{
	// TODO: ���⿡ Ư��ȭ�� �ڵ带 �߰� ��/�Ǵ� �⺻ Ŭ������ ȣ���մϴ�.
	CFile File; // ���� ��ü ����
	CFileDialog SaveDlg(FALSE, _T("raw"), NULL, OFN_HIDEREADONLY);
	// raw ������ �ٸ� �̸����� �����ϱ⸦ ���� ��ȭ���� ��ü ����
	if (SaveDlg.DoModal() == IDOK) {
		// DoModal ��� �Լ����� �����ϱ� ����
		File.Open(SaveDlg.GetPathName(), CFile::modeCreate |
			CFile::modeWrite);
		// ���� ����
		File.Write(m_InputImage, m_size); // ���� ����
		File.Close();
		// ���� �ݱ�
	}
	return TRUE;
}

void CMFCApplicationIPtestDoc::OnFileSave()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	OnSaveDocument(GetPathName());
}


void CMFCApplicationIPtestDoc::OnSumConstant()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CConstantDlg dlg; // ��� ���� �Է¹޴� ��ȭ����
	int i;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	m_OutputImage = new unsigned char[m_Re_size];
	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] + dlg.m_Constant >= 255)
				m_OutputImage[i] = 255;
			// ��� ���� 255���� ũ�� 255 ���
			else
				m_OutputImage[i] = (unsigned char)(m_InputImage[i] + dlg.m_Constant);
			// ��� ���� ȭ�� ������ ����
		}
	}
}



void CMFCApplicationIPtestDoc::OnSubConstant()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CConstantDlg dlg;
	int i;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	m_OutputImage = new unsigned char[m_Re_size];
	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] - dlg.m_Constant < 0)
				m_OutputImage[i] = 0; // ��� ���� 255���� ũ�� 255�� ���
			else
				m_OutputImage[i]
				= (unsigned char)(m_InputImage[i] - dlg.m_Constant);
			// ��� ���� ȭ�� ������ ����
		}
	}
}


void CMFCApplicationIPtestDoc::OnMulConstant()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
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
			// ���� ���� 255���� ũ�� 255�� ���
			else if (m_InputImage[i] * dlg.m_Constant < 0)
				m_OutputImage[i] = 0;
			// ���� ���� 0���� ������ 0�� ���
			else
				m_OutputImage[i]
				= (unsigned char)(m_InputImage[i] * dlg.m_Constant);
			// ��� ���� ȭ�� �� ����
		}
	}
}


void CMFCApplicationIPtestDoc::OnDivConstant()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
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
			// �������� ���� 255���� ũ�� 255�� ���
			else if (m_InputImage[i] / dlg.m_Constant < 0)
				m_OutputImage[i] = 0;
			// �������� ���� 0���� ������ 0�� ���
			else
				m_OutputImage[i]
				= (unsigned char)(m_InputImage[i] / dlg.m_Constant);
			// ��� ���� ȭ�� �� ������
		}
	}
}


void CMFCApplicationIPtestDoc::OnAndOperate()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CConstantDlg dlg;
	int i;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	m_OutputImage = new unsigned char[m_Re_size];
	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			// ��Ʈ ���� AND ����
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
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CConstantDlg dlg;
	int i;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	m_OutputImage = new unsigned char[m_Re_size];
	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			// ��Ʈ ���� OR ����
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
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CConstantDlg dlg;
	int i;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	m_OutputImage = new unsigned char[m_Re_size];
	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			// ��Ʈ ���� XOR ����
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
		m_OutputImage[i] = 255 - m_InputImage[i]; // ���� ������ ����
}



void CMFCApplicationIPtestDoc::OnGammaCorrection()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
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
			// ���� �� ���
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
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CConstantDlg dlg;
	int i;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	m_OutputImage = new unsigned char[m_Re_size];
	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] >= dlg.m_Constant)
				m_OutputImage[i] = 255; // �Ӱ� ������ ũ�� 255 ���
			else
				m_OutputImage[i] = 0; // �Ӱ� ������ ������ 0 ���
		}
	}
}


void CMFCApplicationIPtestDoc::OnStressTransform()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CStressTransformDlg dlg;
	int i;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	m_OutputImage = new unsigned char[m_Re_size];
	if (dlg.DoModal() == IDOK) {
		for (i = 0; i < m_size; i++) {
			// �Է� ���� ���� ���� ���� ���� ���� �� ���̿� ��ġ�ϸ� 255 ���
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
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.

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
			// ���� ������ �ּҰ����� ���� ���� 0
			if (m_InputImage[i] <= MIN)
			{
				m_OutputImage[i] = 0;
			}
			// ���� ������ �ִ밪���� ū ���� 255
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
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	// ������׷��� ���� 0~255
	// ������׷��� ũ�� ���� MAX=255�� ����ȭ�Ͽ� ���
	// ����Ʈ�׷��� ũ�� : 256*256 ����
	int i, j, value;
	unsigned char LOW, HIGH;
	double MAX, MIN, DIF;
	m_Re_height = 256;
	m_Re_width = 256;
	m_Re_size = m_Re_height * m_Re_width;
	LOW = 0;
	HIGH = 255;
	// �ʱ�ȭ
	for (i = 0; i < 256; i++)
		m_HIST[i] = LOW;
	// �� �� ����
	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
	}
	// ����ȭ
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
	// ����ȭ�� ������׷�
	for (i = 0; i < 256; i++)
		m_Scale_HIST[i] = (unsigned char)((m_HIST[i] - MIN) * HIGH / DIF);
	// ����ȭ�� ������׷� ���
	m_OutputImage = new unsigned char[m_Re_size + (256 * 20)];
	for (i = 0; i < m_Re_size; i++)
		m_OutputImage[i] = 255;
	// ����ȭ�� ������׷��� ���� ��� �迭�� ���� ��(0)���� ǥ��
	for (i = 0; i < 256; i++) {
		for (j = 0; j < m_Scale_HIST[i]; j++) {
			m_OutputImage[m_Re_width * (m_Re_height - j - 1) + i] = 0;
		}
	}
	// ������׷��� ����ϰ� �� �Ʒ� �κп� ������׷��� ���� ǥ��
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
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	int i, value;
	unsigned char LOW, HIGH, Temp;
	double SUM = 0.0;
	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;
	LOW = 0;
	HIGH = 255;
	// �ʱ�ȭ
	for (i = 0; i < 256; i++)
		m_HIST[i] = LOW;
	// �� �� ����
	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
	}
	// ���� ������׷� ����
	for (i = 0; i < 256; i++) {
		SUM += m_HIST[i];
		m_Sum_Of_HIST[i] = SUM;
	}
	m_OutputImage = new unsigned char[m_Re_size];
	// �Է� ������ ��Ȱȭ�� �������� ���
	for (i = 0; i < m_size; i++) {
		Temp = m_InputImage[i];
		m_OutputImage[i] = (unsigned char)(m_Sum_Of_HIST[Temp] * HIGH / m_size);
	}
}


void CMFCApplicationIPtestDoc::OnHistoSpec()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
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

	// ���ϴ� ������׷��� �ִ� ������ �Է¹���
	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);
		if (File.GetLength() == (unsigned)m_size) {
			m_DTEMP = new unsigned char[m_size];
			File.Read(m_DTEMP, m_size);
			File.Close();
		}
		else {
			AfxMessageBox(L"Image size not matched");
			// ���� ũ���� ������ ������� ��
			return;
		}
	}
	LOW = 0;
	HIGH = 255;
	// �ʱ�ȭ
	for (i = 0; i < 256; i++) {
		m_HIST[i] = LOW;
		m_DHIST[i] = LOW;
		m_TABLE[i] = LOW;
	}
	// �� �� ����
	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
		Dvalue = (int)m_DTEMP[i];
		m_DHIST[Dvalue]++;
	}
	// ���� ������׷� ����
	for (i = 0; i < 256; i++) {
		SUM += m_HIST[i];
		m_Sum_Of_HIST[i] = SUM;
		DSUM += m_DHIST[i];
		m_Sum_Of_DHIST[i] = DSUM;
	}
	// ���� ������ ��Ȱȭ
	for (i = 0; i < m_size; i++) {
		Temp = m_InputImage[i];
		m_Org_Temp[i] = (unsigned char)(m_Sum_Of_HIST[Temp] * HIGH / m_size);
	}
	// ���� ������׷����� �ּҰ��� �ִ밪 ����
	DMIN = m_Sum_Of_DHIST[0];
	DMAX = m_Sum_Of_DHIST[255];
	// ���ϴ� ������ ��Ȱȭ
	for (i = 0; i < 256; i++) {
		m_Sum_Of_ScHIST[i] = (unsigned char)((m_Sum_Of_DHIST[i]
			- DMIN) * HIGH / (DMAX - DMIN));
	}
	// ������̺��� �̿��� ��ȭ
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
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	int i, value, Dvalue, top, bottom, DADD;
	unsigned char* m_DTEMP = nullptr;  // �ʱ�ȭ
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

	// ���ϴ� ������׷��� �ִ� ������ �Է¹���
	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetFileName(), CFile::modeRead);
		if (File.GetLength() == (unsigned)m_size) {
			m_DTEMP = new unsigned char[m_size];  // �޸� �Ҵ�
			File.Read(m_DTEMP, m_size);
			File.Close();
		}
		else {
			AfxMessageBox(L"Image size not matched");
			delete[] m_DTEMP;  // �޸� ����
			return;
		}
	}
	else {
		AfxMessageBox(L"File selection cancelled");
		delete[] m_DTEMP;  // �޸� ����
		return;
	}

	LOW = 0;
	HIGH = 255;
	// �ʱ�ȭ
	for (i = 0; i < 256; i++) {
		m_HIST[i] = LOW;
		m_DHIST[i] = LOW;
		m_TABLE[i] = LOW;
	}
	// �� �� ����
	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
		Dvalue = (int)m_DTEMP[i];
		m_DHIST[Dvalue]++;
	}
	// ���� ������׷� ����
	for (i = 0; i < 256; i++) {
		SUM += m_HIST[i];
		m_Sum_Of_HIST[i] = SUM;
		DSUM += m_DHIST[i];
		m_Sum_Of_DHIST[i] = DSUM;
	}
	// ���� ������ ��Ȱȭ
	for (i = 0; i < m_size; i++) {
		Temp = m_InputImage[i];
		m_Org_Temp[i] = (unsigned char)(m_Sum_Of_HIST[Temp] * HIGH / m_size);
	}
	// ���� ������׷����� �ּҰ��� �ִ밪 ����
	DMIN = m_Sum_Of_DHIST[0];
	DMAX = m_Sum_Of_DHIST[255];
	// ���ϴ� ������ ��Ȱȭ
	for (i = 0; i < 256; i++) {
		m_Sum_Of_ScHIST[i] = (unsigned char)((m_Sum_Of_DHIST[i] - DMIN) * HIGH / (DMAX - DMIN));
	}
	// ������̺��� �̿��� ��ȭ
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

	// ����� �޸� ����
	delete[] m_DTEMP;
	delete[] m_Org_Temp;
}
*/
