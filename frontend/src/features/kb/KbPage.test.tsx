import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HashRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { KbDocument } from '../../api/types'
import { jsonResponse } from '../chat/testUtils'
import KbPage from './KbPage'

const fetchMock = vi.fn()

const kb = { id: 1, name: '員工手冊', description: '', parser: 'liteparse', vector_weight: null, document_count: 1, created_at: '', updated_at: '' }
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

  it('lists documents, marks unavailable parsers and uploads a picked file right away', async () => {
    window.location.hash = '#/kb/1'
    let uploaded: FormData | null = null
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents' && init?.method === 'POST') {
        uploaded = init.body as FormData
        return Promise.resolve(jsonResponse(doc({ id: 11, status: 'queued' }), 201))
      }
      if (url === '/api/kbs/1/documents')
        return Promise.resolve(jsonResponse(uploaded ? [doc(), doc({ id: 11, filename: 'new.pdf' })] : [doc()]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    expect(await screen.findByText('a.pdf')).toBeInTheDocument()
    expect(screen.getByText('完成')).toBeInTheDocument()
    // options show only the parser's name; its state is a badge next to the select
    expect(screen.getByRole('option', { name: 'LightOnOCR-3（GPU OCR）' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: /無法使用/ })).toBeNull()
    const badge = screen.getByRole('status', { name: '解析器狀態' })
    expect(badge).toHaveTextContent('可使用')
    expect(badge).toHaveAttribute('data-state', 'ok')

    const file = new File(['%PDF-1.4'], 'new.pdf', { type: 'application/pdf' })
    await userEvent.upload(screen.getByLabelText('選擇檔案'), file)
    await waitFor(() => expect(uploaded).not.toBeNull())
    // the knowledge base's setting decides the parser: the upload does not send one
    expect(uploaded!.get('parser')).toBeNull()
    expect((uploaded!.get('file') as File).name).toBe('new.pdf')
    // the document list refreshes with the new file
    expect(await screen.findByRole('cell', { name: 'new.pdf' })).toBeInTheDocument()
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

  it('uploads every file dropped on the upload zone', async () => {
    window.location.hash = '#/kb/1'
    const names: string[] = []
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents' && init?.method === 'POST') {
        names.push(((init.body as FormData).get('file') as File).name)
        return Promise.resolve(jsonResponse(doc({ id: 10 + names.length }), 201))
      }
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    const zone = await screen.findByRole('button', { name: /拖曳檔案到這裡/ })
    const files = [new File(['%PDF'], 'a.pdf'), new File(['# 標題'], 'b.md'), new File(['x'], 'c.png')]
    fireEvent.drop(zone, { dataTransfer: { files, types: ['Files'] } })
    await waitFor(() => expect(names).toEqual(['a.pdf', 'b.md', 'c.png']))
  })

  it('shows the upload error from the server next to the file', async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents' && init?.method === 'POST')
        return Promise.resolve(jsonResponse({ detail: '無法讀取檔案內容' }, 422))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    await userEvent.upload(await screen.findByLabelText('選擇檔案'), new File(['x'], 'a.txt'))
    const row = await screen.findByRole('listitem', { name: 'a.txt' })
    expect(await within(row).findByText('無法讀取檔案內容')).toBeInTheDocument()
  })

  it('runs a retrieval test', async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc()]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      if (url === '/api/search-settings') return Promise.resolve(jsonResponse({ vector_weight: 0.5, top_k: 5 }))
      if (url.startsWith('/api/kbs/1/search?q='))
        return Promise.resolve(
          jsonResponse([{ document_id: 10, filename: 'a.pdf', page: 2, score: 0.8123, vector_score: 0.81234, keyword_score: null, text: '特休十四天' }]),
        )
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    await userEvent.type(await screen.findByRole('textbox', { name: '檢索問題' }), '特休')
    await userEvent.click(screen.getByRole('button', { name: '檢索' }))
    expect(await screen.findByText('特休十四天')).toBeInTheDocument()
    expect(screen.getByText('[1] a.pdf · 第 2 頁')).toBeInTheDocument()
    expect(screen.getByText(/^相關度/).textContent).toBe('相關度 0.8123　向量 0.812　關鍵字 —')
    expect(screen.queryByText(/相似度/)).not.toBeInTheDocument()
    expect(
      fetchMock.mock.calls.some((c) => String(c[0]) === '/api/kbs/1/search?q=%E7%89%B9%E4%BC%91&vector_weight=0.5'),
    ).toBe(true)
  })

  it('tries other hybrid weights in the retrieval test', async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc()]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      if (url === '/api/search-settings') return Promise.resolve(jsonResponse({ vector_weight: 0.6, top_k: 5 }))
      if (url.startsWith('/api/kbs/1/search?q=')) return Promise.resolve(jsonResponse([]))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    const slider = await screen.findByRole('slider', { name: '向量權重' })
    await waitFor(() => expect(slider).toHaveValue('0.6'))
    expect(screen.getByText('向量 0.60 ／ 關鍵字 0.40（知識庫目前的權重）')).toBeInTheDocument()

    fireEvent.change(slider, { target: { value: '0.3' } })
    expect(screen.getByText('向量 0.30 ／ 關鍵字 0.70')).toBeInTheDocument()
    await userEvent.type(screen.getByRole('textbox', { name: '檢索問題' }), 'q')
    await userEvent.click(screen.getByRole('button', { name: '檢索' }))
    await screen.findByText('沒有找到相關段落')
    expect(fetchMock.mock.calls.some((c) => String(c[0]) === '/api/kbs/1/search?q=q&vector_weight=0.3')).toBe(true)

    await userEvent.click(screen.getByRole('button', { name: '還原' }))
    expect(slider).toHaveValue('0.6')
  })

  it('searches with the server default when the weight settings cannot be loaded', async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc()]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      if (url.startsWith('/api/kbs/1/search?q=')) return Promise.resolve(jsonResponse([]))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    await userEvent.type(await screen.findByRole('textbox', { name: '檢索問題' }), 'q')
    await userEvent.click(screen.getByRole('button', { name: '檢索' }))
    await screen.findByText('沒有找到相關段落')
    expect(fetchMock.mock.calls.some((c) => String(c[0]) === '/api/kbs/1/search?q=q')).toBe(true)
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
  })

  it('opens the chunk viewer of a document and comes back', async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc(), doc({ id: 11, filename: 'b.pdf', status: 'parsing' })]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      if (url === '/api/search-settings') return Promise.resolve(jsonResponse({ vector_weight: 0.5, top_k: 5 }))
      if (url === '/api/kbs/1/documents/10/chunks')
        return Promise.resolve(
          jsonResponse({ document: doc(), chunks: [{ index: 1, page: 1, kind: 'text', text: '第一塊內容', chars: 5, images: [], page_image: null }] }),
        )
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    await userEvent.click(await screen.findByRole('button', { name: '查看切塊 a.pdf' }))
    await waitFor(() => expect(window.location.hash).toBe('#/kb/1/documents/10'))
    expect(await screen.findByText('第一塊內容')).toBeInTheDocument()
    // the knowledge base stays selected in the list
    expect(screen.getByRole('button', { name: /員工手冊/, current: 'page' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '返回 員工手冊' }))
    await waitFor(() => expect(window.location.hash).toBe('#/kb/1'))
    expect(await screen.findByRole('heading', { name: '員工手冊' })).toBeInTheDocument()
  })

  it('opens the chunk viewer straight from its address', async () => {
    window.location.hash = '#/kb/1/documents/10'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents/10/chunks')
        return Promise.resolve(jsonResponse({ document: doc(), chunks: [] }))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    expect(await screen.findByRole('heading', { name: 'a.pdf' })).toBeInTheDocument()
    expect(window.location.hash).toBe('#/kb/1/documents/10')
  })

  it('only offers the chunk viewer for documents that have chunks', async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc({ id: 11, filename: 'b.pdf', status: 'parsing' })]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    expect(await screen.findByText('b.pdf')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '查看切塊 b.pdf' })).toBeNull()
  })

  it("shows the knowledge base's parser setting, which survives a reload", async () => {
    window.location.hash = '#/kb/1'
    const ocrKb = { ...kb, parser: 'lightonocr' }
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([ocrKb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc({ parser: 'lightonocr' })]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    const select = await screen.findByRole('combobox', { name: '知識庫的解析器' })
    await waitFor(() => expect(select).toHaveValue('lightonocr'))
    const badge = screen.getByRole('status', { name: '解析器狀態' })
    expect(badge).toHaveAttribute('data-state', 'unavailable')
    expect(badge).toHaveTextContent('目前無法使用：未安裝 OCR 套件')
    expect(screen.getByText(/只影響之後上傳的 PDF 與圖片/)).toBeInTheDocument()
    expect(screen.getByText(/PDF 與圖片會用「LightOnOCR-3」解析/)).toBeInTheDocument()
  })

  it('refreshes the parser badge while the page stays open', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    window.location.hash = '#/kb/1'
    const ocrKb = { ...kb, parser: 'lightonocr' }
    let ocrReady = false
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([ocrKb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc()]))
      if (url === '/api/parsers')
        return Promise.resolve(jsonResponse(parsers.map((p) => ({ ...p, available: p.available || ocrReady }))))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    const badge = await screen.findByRole('status', { name: '解析器狀態' })
    await waitFor(() => expect(badge).toHaveAttribute('data-state', 'unavailable'))
    // vLLM finishes loading: the next refresh turns the badge green
    ocrReady = true
    await act(() => vi.advanceTimersByTimeAsync(15000))
    await waitFor(() => expect(badge).toHaveAttribute('data-state', 'ok'))
    // a failed refresh keeps the last known state
    fetchMock.mockImplementation((url: string) =>
      url === '/api/parsers' ? Promise.reject(new Error('offline')) : Promise.resolve(jsonResponse([])),
    )
    await act(() => vi.advanceTimersByTimeAsync(15000))
    expect(badge).toHaveAttribute('data-state', 'ok')
  })

  it('saves a new parser setting for the knowledge base', async () => {
    window.location.hash = '#/kb/1'
    let patched: unknown = null
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs/1' && init?.method === 'PATCH') {
        patched = JSON.parse(init.body as string)
        return Promise.resolve(jsonResponse({ ...kb, parser: 'lightonocr' }))
      }
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([doc()]))
      if (url === '/api/parsers')
        return Promise.resolve(jsonResponse(parsers.map((p) => ({ ...p, available: true }))))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    const select = await screen.findByRole('combobox', { name: '知識庫的解析器' })
    await waitFor(() => expect(select).toHaveValue('liteparse'))
    await userEvent.selectOptions(select, 'lightonocr')
    await waitFor(() => expect(patched).toEqual({ parser: 'lightonocr' }))
    await waitFor(() => expect(select).toHaveValue('lightonocr'))
    // documents already uploaded keep the parser they were processed with
    expect(screen.getByRole('cell', { name: 'LiteParse' })).toBeInTheDocument()
  })

  it('shows why the parser setting could not be saved', async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs/1' && init?.method === 'PATCH')
        return Promise.resolve(jsonResponse({ detail: '未知的解析器' }, 422))
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    const select = await screen.findByRole('combobox', { name: '知識庫的解析器' })
    await waitFor(() => expect(select).not.toBeDisabled())
    await userEvent.selectOptions(select, 'lightonocr')
    expect(await screen.findByRole('alert')).toHaveTextContent('未知的解析器')
    expect(select).toHaveValue('liteparse')
  })

  it('shows the parser being saved and locks the setting while files upload', async () => {
    window.location.hash = '#/kb/1'
    let finishSave!: () => void
    let finishUpload!: () => void
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs/1' && init?.method === 'PATCH')
        return new Promise((resolve) => {
          finishSave = () => resolve(jsonResponse({ ...kb, parser: 'lightonocr' }))
        })
      if (url === '/api/kbs/1/documents' && init?.method === 'POST')
        return new Promise((resolve) => {
          finishUpload = () => resolve(jsonResponse(doc({ id: 12 }), 201))
        })
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([]))
      if (url === '/api/parsers')
        return Promise.resolve(jsonResponse(parsers.map((p) => ({ ...p, available: true }))))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    const select = await screen.findByRole('combobox', { name: '知識庫的解析器' })
    await waitFor(() => expect(select).not.toBeDisabled())
    await userEvent.selectOptions(select, 'lightonocr')
    // the new choice shows while it is being saved, not the old one
    expect(select).toHaveValue('lightonocr')
    expect(select).toBeDisabled()
    finishSave()
    await waitFor(() => expect(select).not.toBeDisabled())
    expect(select).toHaveValue('lightonocr')

    await userEvent.upload(screen.getByLabelText('選擇檔案'), new File(['%PDF'], 'c.pdf'))
    await waitFor(() => expect(select).toBeDisabled())
    finishUpload()
    await waitFor(() => expect(select).not.toBeDisabled())
  })

  it("shows the knowledge base's own weight, which the chat uses", async () => {
    window.location.hash = '#/kb/1'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([{ ...kb, vector_weight: 0.8 }]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      if (url === '/api/search-settings') return Promise.resolve(jsonResponse({ vector_weight: 0.5, top_k: 5 }))
      if (url.startsWith('/api/kbs/1/search?q=')) return Promise.resolve(jsonResponse([]))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    expect((await screen.findByText(/^混合查詢權重：/)).textContent).toBe('混合查詢權重：向量 0.80 ／ 關鍵字 0.20（自訂）')
    const slider = screen.getByRole('slider', { name: '向量權重' })
    expect(slider).toHaveValue('0.8')
    await userEvent.type(screen.getByRole('textbox', { name: '檢索問題' }), 'q')
    await userEvent.click(screen.getByRole('button', { name: '檢索' }))
    await screen.findByText('沒有找到相關段落')
    expect(fetchMock.mock.calls.some((c) => String(c[0]) === '/api/kbs/1/search?q=q&vector_weight=0.8')).toBe(true)
  })

  it('saves a weight tried in the retrieval test, and goes back to the system default', async () => {
    window.location.hash = '#/kb/1'
    const patches: unknown[] = []
    let current: number | null = null
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs/1' && init?.method === 'PATCH') {
        const body = JSON.parse(init.body as string)
        patches.push(body)
        current = body.vector_weight
        return Promise.resolve(jsonResponse({ ...kb, vector_weight: current }))
      }
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([kb]))
      if (url === '/api/kbs/1/documents') return Promise.resolve(jsonResponse([]))
      if (url === '/api/parsers') return Promise.resolve(jsonResponse(parsers))
      if (url === '/api/search-settings') return Promise.resolve(jsonResponse({ vector_weight: 0.5, top_k: 5 }))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    expect((await screen.findByText(/^混合查詢權重：/)).textContent).toBe('混合查詢權重：向量 0.50 ／ 關鍵字 0.50（系統預設）')
    expect(screen.queryByRole('button', { name: '改回系統預設' })).toBeNull()

    fireEvent.change(screen.getByRole('slider', { name: '向量權重' }), { target: { value: '0.3' } })
    await userEvent.click(screen.getByRole('button', { name: '設為這個知識庫的權重' }))
    await waitFor(() => expect(patches).toEqual([{ vector_weight: 0.3 }]))
    await waitFor(() =>
      expect(screen.getByText(/^混合查詢權重：/).textContent).toBe('混合查詢權重：向量 0.30 ／ 關鍵字 0.70（自訂）'),
    )
    expect(screen.getByText('向量 0.30 ／ 關鍵字 0.70（知識庫目前的權重）')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: '改回系統預設' }))
    await waitFor(() => expect(patches).toEqual([{ vector_weight: 0.3 }, { vector_weight: null }]))
    await waitFor(() => expect(screen.getByRole('slider', { name: '向量權重' })).toHaveValue('0.5'))
  })
})
