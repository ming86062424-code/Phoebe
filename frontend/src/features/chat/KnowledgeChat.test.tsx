import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HashRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Source } from '../../api/types'
import { NavGuardProvider } from '../../shell/navGuard'
import ChatPage from './ChatPage'
import MessageBubble from './MessageBubble'
import { applyEvent, fromMessages, type ChatItem } from './chatState'
import { conversation, jsonResponse, ndjsonResponse } from './testUtils'

const fetchMock = vi.fn()

const source: Source = {
  document_id: 1,
  filename: '手冊.pdf',
  page: 3,
  score: 0.9,
  text: '員工每年享有特休假十四天',
}

const kbs = [
  { id: 7, name: '員工手冊', description: '', parser: 'liteparse', document_count: 1, created_at: '', updated_at: '' },
]

function setup() {
  return render(
    <HashRouter>
      <NavGuardProvider>
        <Routes>
          <Route path="/chat/*" element={<ChatPage />} />
        </Routes>
      </NavGuardProvider>
    </HashRouter>,
  )
}

function calls(): [string, RequestInit | undefined][] {
  return fetchMock.mock.calls.map((c) => [String(c[0]), c[1] as RequestInit | undefined])
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('sources in chat state', () => {
  it('stores sources from the stream and from saved messages', () => {
    const fresh: ChatItem = { key: 'a', role: 'assistant', content: '', thinking: '', status: 'streaming' }
    expect(applyEvent(fresh, { type: 'sources', sources: [source] }, 0).sources).toEqual([source])
    const [loaded] = fromMessages([
      { id: 1, role: 'assistant', content: 'x', sources: [source], created_at: '' },
    ])
    expect(loaded.sources).toEqual([source])
  })
})

describe('MessageBubble sources', () => {
  const base: ChatItem = { key: 'a', role: 'assistant', content: '答', thinking: '', status: 'done' }

  it('expands to show the cited pages', async () => {
    render(<MessageBubble item={{ ...base, sources: [source] }} kbId={7} />)
    expect(screen.queryByText(/手冊\.pdf/)).toBeNull()
    await userEvent.click(screen.getByRole('button', { name: '參考資料（1）' }))
    expect(screen.getByText('[1] 手冊.pdf · 第 3 頁')).toBeInTheDocument()
    expect(screen.getByText('員工每年享有特休假十四天')).toBeInTheDocument()
  })

  it('shows the cards by score with their original numbers and images from the chat knowledge base', async () => {
    const sources: Source[] = [
      { ...source, filename: '低.pdf', score: 0.01 },
      {
        ...source,
        document_id: 17,
        filename: '高.pdf',
        score: 0.03,
        page_image: { name: 'page_3.png', width: 10, height: 10 },
      },
    ]
    render(<MessageBubble item={{ ...base, sources }} kbId={7} />)
    await userEvent.click(screen.getByRole('button', { name: '參考資料（2）' }))
    const items = screen.getAllByRole('listitem')
    expect(items[0]).toHaveTextContent('[2] 高.pdf · 第 3 頁')
    expect(items[1]).toHaveTextContent('[1] 低.pdf · 第 3 頁')
    await userEvent.click(screen.getAllByRole('button', { name: '展開' })[0])
    expect(screen.getByRole('img', { name: '第 3 頁' })).toHaveAttribute(
      'src',
      '/api/kbs/7/documents/17/images/page_3.png',
    )
  })

  it('says when the knowledge base had nothing relevant', () => {
    render(<MessageBubble item={{ ...base, sources: [] }} />)
    expect(screen.getByText('知識庫中沒有找到相關內容')).toBeInTheDocument()
  })
})

describe('ChatPage knowledge base selector', () => {
  it('applies the chosen knowledge base to a new conversation before sending', async () => {
    window.location.hash = '#/chat'
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse(kbs))
      if (url === '/api/conversations' && init?.method === 'POST')
        return Promise.resolve(jsonResponse({ ...conversation(5), messages: undefined }, 201))
      if (url === '/api/conversations') return Promise.resolve(jsonResponse([]))
      if (url === '/api/conversations/5' && init?.method === 'PATCH')
        return Promise.resolve(jsonResponse({ ...conversation(5), kb_id: 7 }))
      if (url === '/api/conversations/5/messages')
        return Promise.resolve(
          ndjsonResponse([
            { type: 'sources', sources: [source] },
            { type: 'content', text: '十四天' },
            { type: 'done', message_id: 2 },
          ]),
        )
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    await userEvent.selectOptions(await screen.findByRole('combobox', { name: '知識庫' }), '7')
    await userEvent.type(screen.getByRole('textbox'), '特休幾天')
    await userEvent.click(screen.getByRole('button', { name: '送出' }))
    expect(await screen.findByRole('button', { name: '參考資料（1）' })).toBeInTheDocument()
    const order = calls().map(([u, i]) => `${i?.method ?? 'GET'} ${u}`)
    const patch = order.indexOf('PATCH /api/conversations/5')
    expect(patch).toBeGreaterThan(-1)
    expect(patch).toBeLessThan(order.indexOf('POST /api/conversations/5/messages'))
    expect(JSON.parse(String(calls()[order.indexOf('PATCH /api/conversations/5')][1]?.body))).toEqual({
      kb_id: 7,
    })
  })

  it('shows the saved knowledge base of an existing conversation and updates it', async () => {
    window.location.hash = '#/chat/3'
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse(kbs))
      if (url === '/api/conversations') return Promise.resolve(jsonResponse([]))
      if (url === '/api/conversations/3' && init?.method === 'PATCH')
        return Promise.resolve(jsonResponse({ ...conversation(3), kb_id: null }))
      if (url === '/api/conversations/3')
        return Promise.resolve(jsonResponse({ ...conversation(3), kb_id: 7 }))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    const select = await screen.findByRole('combobox', { name: '知識庫' })
    await waitFor(() => expect(select).toHaveValue('7'))
    await userEvent.selectOptions(select, '')
    await waitFor(() => expect(select).toHaveValue(''))
    expect(calls().some(([u, i]) => u === '/api/conversations/3' && i?.method === 'PATCH')).toBe(true)
  })

  it('hides the selector when the user has no knowledge bases', async () => {
    window.location.hash = '#/chat'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse([]))
      if (url === '/api/conversations') return Promise.resolve(jsonResponse([]))
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    await waitFor(() => expect(calls().some(([u]) => u === '/api/kbs')).toBe(true))
    expect(screen.queryByRole('combobox', { name: '知識庫' })).toBeNull()
  })
})
