import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HashRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { KbDocument } from '../../api/types'
import { jsonResponse } from '../chat/testUtils'
import KbPage from './KbPage'

const fetchMock = vi.fn()

const kb = { id: 1, name: '員工手冊', description: '', document_count: 1, created_at: '', updated_at: '' }
const parsers = [
  { id: 'liteparse', label: 'LiteParse（快速）', available: true, reason: '' },
  { id: 'lightonocr', label: 'LightOnOCR-3（GPU OCR）', available: false, reason: '未安裝 OCR 套件' },
]

function doc(overrides: Partial<KbDocument> = {}): KbDocument {
  return {
    id: 10,
    kb_id: 1,
    filename: 'a.pdf',
    parser: 'liteparse',
    status: 'ready',
    error: null,
    progress_done: 2,
    progress_total: 2,
    chunk_count: 4,
    size_bytes: 2048,
    created_at: '',
    updated_at: '',
    ...overrides,
  }
}

function setup() {
  return render(
    <HashRouter>
      <Routes>
        <Route path="/kb/*" element={<KbPage />} />
      </Routes>
    </HashRouter>,
  )
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('KbPage', () => {
  it('shows an empty state and creates a knowledge base', async () => {
    window.location.hash = '#/kb'
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs' && init?.method === 'POST')
        return Promise.resolve(jsonResponse({ ...kb, document_count: 0 }, 201))
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    expect(await screen.findByText(/建立第一個知識庫/)).toBeInTheDocument()
    await userEvent.type(screen.getByRole('textbox', { name: '新知識庫名稱' }), '員工手冊')
    await userEvent.click(screen.getByRole('button', { name: '新增知識庫' }))
    await waitFor(() => expect(window.location.hash).toBe('#/kb/1'))
    expect(await screen.findByRole('heading', { name: '員工手冊' })).toBeInTheDocument()
    expect(await screen.findByText('尚未上傳文件')).toBeInTheDocument()
  })

  it('lists documents, marks unavailable parsers and uploads with the chosen parser', async () => {
    window.location.hash = '#/kb/1'
    let uploaded: FormData | null = null
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents' && init?.method === 'POST') {
        uploaded = init.body as FormData
        return Promise.resolve(jsonResponse(doc({ id: 11, status: 'ready' }), 201))
      }
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc()]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    expect(await screen.findByText('a.pdf')).toBeInTheDocument()
    expect(screen.getByText('完成')).toBeInTheDocument()
    expect(await screen.findByText(/未安裝 OCR 套件/)).toBeInTheDocument()
    expect(screen.getByRole('option', { name: /LightOnOCR-3.*不可用/ })).toBeDisabled()

    const file = new File(['%PDF-1.4'], 'new.pdf', { type: 'application/pdf' })
    await userEvent.upload(screen.getByLabelText('選擇 PDF 檔案'), file)
    await userEvent.click(screen.getByRole('button', { name: '上傳' }))
    await waitFor(() => expect(uploaded).not.toBeNull())
    expect(uploaded!.get('parser')).toBe('liteparse')
    expect((uploaded!.get('file') as File).name).toBe('new.pdf')
  })

  it('shows page progress and failure details', async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents')
        return Promise.resolve(
          jsonResponse([
            doc({ id: 1, filename: 'scan.pdf', status: 'parsing', progress_done: 3, progress_total: 9 }),
            doc({ id: 2, filename: 'bad.pdf', status: 'failed', error: '沒有擷取到任何文字' }),
          ]),
        )
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    expect(await screen.findByText('解析中 3/9 頁')).toBeInTheDocument()
    expect(screen.getByText('沒有擷取到任何文字')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '重試 bad.pdf' })).toBeInTheDocument()
  })

  it('runs a retrieval test', async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc()]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      if (url.startsWith('/api/kbs/1/search?q='))
        return Promise.resolve(
          jsonResponse([{ document_id: 10, filename: 'a.pdf', page: 2, score: 0.81234, text: '特休十四天' }]),
        )
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    await userEvent.type(await screen.findByRole('textbox', { name: '檢索問題' }), '特休')
    await userEvent.click(screen.getByRole('button', { name: '檢索' }))
    expect(await screen.findByText('特休十四天')).toBeInTheDocument()
    expect(screen.getByText('相似度 0.812')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some((c) => String(c[0]) === '/api/kbs/1/search?q=%E7%89%B9%E4%BC%91')).toBe(true)
  })
})
