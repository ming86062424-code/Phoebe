import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HashRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { NavGuardProvider } from '../../shell/navGuard'
import ChatPage from './ChatPage'
import { conversation, heldResponse, jsonResponse } from './testUtils'

const fetchMock = vi.fn()

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

function urls(): string[] {
  return fetchMock.mock.calls.map((c) => String(c[0]))
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  try {
    localStorage.clear()
  } catch {
    // ignore
  }
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ChatPage', () => {
  it('retry button disappears while streaming', async () => {
    const held = heldResponse()
    window.location.hash = '#/chat/1'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/conversations') return Promise.resolve(jsonResponse([]))
      if (url === '/api/conversations/1')
        return Promise.resolve(jsonResponse(conversation(1, [{ role: 'user', content: 'q' }])))
      if (url === '/api/conversations/1/regenerate') return Promise.resolve(held.response)
      return Promise.reject(new Error('unexpected ' + url))
    })
    setup()
    const retry = await screen.findByRole('button', { name: '重新送出' })
    await userEvent.click(retry)
    await waitFor(() => expect(screen.queryByRole('button', { name: '重新送出' })).toBeNull())
    expect(urls().filter((u) => u.endsWith('/regenerate'))).toHaveLength(1)
    held.push({ type: 'done', message_id: 1 })
    held.close()
    await waitFor(() => expect(screen.queryByText('回覆產生中', { exact: false })).toBeNull())
  })

  it('ChatPage redirects to #/chat when conversation is 404', async () => {
    window.location.hash = '#/chat/999'
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/conversations') return Promise.resolve(jsonResponse([]))
      return Promise.resolve(jsonResponse({ detail: 'nope' }, 404))
    })
    setup()
    await waitFor(() => expect(window.location.hash).toBe('#/chat'))
    expect(await screen.findByText('找不到這個對話')).toBeInTheDocument()
  })

  it('conversation list collapses and persists', async () => {
    window.location.hash = '#/chat'
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        jsonResponse([
          { id: 1, title: '', created_at: '', updated_at: '' },
        ]),
      ),
    )
    const { unmount } = setup()
    expect(await screen.findByRole('navigation', { name: '對話列表' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '收合對話列表' }))
    expect(localStorage.getItem('phoebe.chat.listCollapsed')).toBe('true')
    expect(screen.queryByRole('navigation', { name: '對話列表' })).toBeNull()
    unmount()
    setup()
    expect(screen.getByRole('button', { name: '展開對話列表' })).toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: '對話列表' })).toBeNull()
  })
})
