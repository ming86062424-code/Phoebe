import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { DocumentChunk, KbDocument } from '../../api/types'
import { jsonResponse } from '../chat/testUtils'
import ChunkViewer from './ChunkViewer'

const fetchMock = vi.fn()
const kb = { id: 1, name: '員工手冊', description: '', document_count: 1, created_at: '', updated_at: '' }

function doc(overrides: Partial<KbDocument> = {}): KbDocument {
  return {
    id: 5, kb_id: 1, filename: '手冊.pdf', parser: 'liteparse', status: 'ready', error: null,
    progress_done: 3, progress_total: 3, chunk_count: 3, size_bytes: 2048, created_at: '', updated_at: '',
    ...overrides,
  }
}

function chunk(overrides: Partial<DocumentChunk> = {}): DocumentChunk {
  return { index: 1, page: 1, kind: 'text', text: '第一段', chars: 3, images: [], page_image: null, ...overrides }
}

function setup(document: KbDocument, chunks: DocumentChunk[], onBack = vi.fn()) {
  fetchMock.mockImplementation((url: string) => {
    if (url === '/api/kbs/1/documents/5/chunks') return Promise.resolve(jsonResponse({ document, chunks }))
    return Promise.reject(new Error('unexpected ' + url))
  })
  render(<ChunkViewer kb={kb} documentId={5} onBack={onBack} />)
  return onBack
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => vi.unstubAllGlobals())

describe('ChunkViewer', () => {
  it('lists every chunk in order with its page, kind and length', async () => {
    setup(doc(), [
      chunk({ index: 1, page: 1, kind: 'page_image', text: '封面文字', chars: 4 }),
      chunk({ index: 2, page: 2, kind: 'page_image', text: '第二頁內容', chars: 5 }),
      chunk({ index: 3, page: 3, kind: 'page_image', text: '第三頁', chars: 3 }),
    ])
    expect(await screen.findByRole('heading', { name: '手冊.pdf' })).toBeInTheDocument()
    const items = screen.getAllByRole('article')
    expect(items).toHaveLength(3)
    expect(within(items[0]).getByText('#1 · 第 1 頁 · 整頁圖＋文字 · 4 字')).toBeInTheDocument()
    expect(within(items[1]).getByText('第二頁內容')).toBeInTheDocument()
    expect(screen.getByText(/^共 3 塊/).textContent).toBe('共 3 塊　總字數 12　平均 4 字　最長 5 字')
  })

  it('explains how each parser splits and embeds the document', async () => {
    setup(doc({ parser: 'lightonocr' }), [chunk()])
    expect((await screen.findByText(/^解析器：/)).textContent).toBe('解析器：LightOnOCR　狀態：完成')
    expect(screen.getByText(/每頁一塊.*OCR 整理出的文字/)).toBeInTheDocument()
  })

  it('labels text-file chunks as full text', async () => {
    setup(doc({ filename: 'faq.md', parser: 'text' }), [
      chunk({ index: 1, page: 0, text: '# 常見問題', chars: 6 }),
    ])
    expect(await screen.findByText('#1 · 全文 · 文字 · 6 字')).toBeInTheDocument()
    expect(screen.getByText(/依標題分段/)).toBeInTheDocument()
  })

  it('shows the pictures of a chunk', async () => {
    setup(doc(), [
      chunk({
        kind: 'page_image',
        text: '說明 ![圖片 1](img:img_1_1.png) 地圖',
        images: [{ name: 'img_1_1.png', width: 3, height: 4 }],
        page_image: { name: 'page_1.png', width: 6, height: 8 },
      }),
    ])
    const inline = await screen.findByRole('img', { name: '圖片 1 地圖' })
    expect(inline).toHaveAttribute('src', '/api/kbs/1/documents/5/images/img_1_1.png')
    expect(screen.getByRole('img', { name: '第 1 頁' })).toHaveAttribute(
      'src',
      '/api/kbs/1/documents/5/images/page_1.png',
    )
  })

  it('tells why there are no chunks yet', async () => {
    setup(doc({ status: 'parsing', chunk_count: 0 }), [])
    expect(await screen.findByText('這份文件還沒有處理完成（解析中），完成後才會有切塊。')).toBeInTheDocument()
  })

  it('shows the failure reason of a failed document', async () => {
    setup(doc({ status: 'failed', error: '沒有擷取到任何文字' }), [])
    expect(await screen.findByText(/沒有擷取到任何文字/)).toBeInTheDocument()
  })

  it('shows a load error', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: '無法讀取切塊：無法連線到 Milvus' }, 503))
    render(<ChunkViewer kb={kb} documentId={5} onBack={vi.fn()} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('無法讀取切塊：無法連線到 Milvus')
  })

  it('goes back to the knowledge base', async () => {
    const onBack = setup(doc(), [chunk()])
    await userEvent.click(await screen.findByRole('button', { name: '返回 員工手冊' }))
    expect(onBack).toHaveBeenCalled()
  })
})
