import { act, render, renderHook, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { HashRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { NavGuardProvider } from '../../shell/navGuard'
import ChatPage from './ChatPage'
import { conversation, jsonResponse, ndjsonResponse } from './testUtils'
import { useChat } from './useChat'

const fetchMock = vi.fn()

const kbs = [
  { id: 7, name: '員工手冊', description: '', parser: 'liteparse', vector_weight: null, document_count: 1, created_at: '', updated_at: '' },
]

function renderPage() {
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

function renderChat(id: number | null) {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <NavGuardProvider>{children}</NavGuardProvider>
  )
  return renderHook(
    (props: { id: number | null }) =>
      useChat({ conversationId: props.id, onConversationCreated: vi.fn(), onFinished: vi.fn() }),
    { wrapper, initialProps: { id } },
  )
}

function patches(): string[] {
  return fetchMock.mock.calls
    .filter((c) => (c[1] as RequestInit | undefined)?.method === 'PATCH')
    .map((c) => String(c[0]))
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('knowledge base selection review fixes', () => {
  it('#2 a failed list load shows an error and leaves the conversation setting alone', async () => {
    window.location.hash = '#/chat/3'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse({ detail: '壞了' }, 500))
      if (url === '/api/conversations') return Promise.resolve(jsonResponse([]))
      if (url === '/api/conversations/3')
        return Promise.resolve(jsonResponse({ ...conversation(3), kb_id: 7 }))
      return Promise.reject(new Error('unexpected ' + url))
    })
    renderPage()
    expect(await screen.findByText('無法載入知識庫列表')).toBeInTheDocument()
    await act(async () => {})
    expect(patches()).toEqual([])
  })

  it('#2 a knowledge base missing from the list is marked, not cleared', async () => {
    window.location.hash = '#/chat/3'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/kbs') return Promise.resolve(jsonResponse(kbs))
      if (url === '/api/conversations') return Promise.resolve(jsonResponse([]))
      if (url === '/api/conversations/3')
        return Promise.resolve(jsonResponse({ ...conversation(3), kb_id: 99 }))
      return Promise.reject(new Error('unexpected ' + url))
    })
    renderPage()
    const select = await screen.findByRole('combobox', { name: '知識庫' })
    await waitFor(() => expect(select).toHaveValue('99'))
    expect(screen.getByRole('option', { name: '（知識庫不存在）' })).toBeInTheDocument()
    expect(patches()).toEqual([])
  })

  it('#6 switching to a new chat does not carry over the previous knowledge base', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/conversations/3')
        return Promise.resolve(jsonResponse({ ...conversation(3), kb_id: 7 }))
      return Promise.reject(new Error('unexpected ' + url))
    })
    const { result, rerender } = renderChat(3)
    await waitFor(() => expect(result.current.kbId).toBe(7))
    rerender({ id: null })
    expect(result.current.kbId).toBeNull()
  })

  it('#7 a failed knowledge base PATCH still sends the message and reports the error', async () => {
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/conversations' && init?.method === 'POST')
        return Promise.resolve(jsonResponse({ ...conversation(9), messages: undefined }, 201))
      if (url === '/api/conversations/9' && init?.method === 'PATCH')
        return Promise.resolve(jsonResponse({ detail: '找不到知識庫' }, 404))
      if (url === '/api/conversations/9/messages')
        return Promise.resolve(
          ndjsonResponse([
            { type: 'content', text: '好' },
            { type: 'done', message_id: 2 },
          ]),
        )
      return Promise.reject(new Error('unexpected ' + url))
    })
    const { result } = renderChat(null)
    await act(() => result.current.selectKb(7))
    let rejected: string | null = 'not called'
    await act(async () => {
      rejected = await result.current.send('x')
    })
    expect(rejected).toBeNull()
    expect(result.current.items.map((i) => i.content)).toEqual(['x', '好'])
    expect(result.current.kbId).toBeNull()
    expect(result.current.kbError).toBe('找不到知識庫')
  })
})
