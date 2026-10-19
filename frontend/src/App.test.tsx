import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

function mockMe(status: number, isAdmin = false) {
  const json = (data: unknown, st = 200) => new Response(JSON.stringify(data), { status: st })
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((url: string) => {
      if (url === '/api/me') {
        return Promise.resolve(
          status === 200 ? json({ id: 1, username: 'amy', is_admin: isAdmin }) : json({ detail: '未登入' }, status),
        )
      }
      if (url === '/api/conversations') return Promise.resolve(json([]))
      return Promise.resolve(
        json({ id: 12, title: 't', created_at: '', updated_at: '', messages: [] }),
      )
    }),
  )
}

describe('App', () => {
  beforeEach(() => {
    window.location.hash = '#/chat'
  })

  it('redirects to #/login when /api/me is 401', async () => {
    mockMe(401)
    render(<App />)
    expect(await screen.findByRole('button', { name: '登入' })).toBeInTheDocument()
    expect(window.location.hash).toBe('#/login')
  })

  it('shows the shell when logged in', async () => {
    mockMe(200)
    render(<App />)
    expect(await screen.findByText('amy')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'AI 對話' })).toBeInTheDocument()
  })

  it('non-admin is redirected from settings', async () => {
    mockMe(200)
    window.location.hash = '#/settings'
    render(<App />)
    await screen.findByText('amy')
    await vi.waitFor(() => expect(window.location.hash).toBe('#/chat'))
    expect(screen.queryByRole('link', { name: '系統設定' })).not.toBeInTheDocument()
  })

  it('admin can open settings', async () => {
    mockMe(200, true)
    window.location.hash = '#/settings'
    render(<App />)
    await screen.findByText('amy')
    expect(window.location.hash).toBe('#/settings')
    expect(screen.getByRole('link', { name: '系統設定' })).toBeInTheDocument()
  })

  it('redirects / to /chat', async () => {
    mockMe(200)
    window.location.hash = '#/'
    render(<App />)
    await screen.findByText('amy')
    expect(window.location.hash).toBe('#/chat')
  })

  it('shows an error with retry when /api/me fails with a non-401 error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network')))
    render(<App />)
    expect(await screen.findByText('無法連線到伺服器')).toBeInTheDocument()
    mockMe(200)
    await userEvent.click(screen.getByRole('button', { name: '重試' }))
    expect(await screen.findByText('amy')).toBeInTheDocument()
  })

  it('does not remount the shell when moving from /chat to /chat/12', async () => {
    mockMe(200)
    render(<App />)
    const link = await screen.findByRole('link', { name: 'AI 對話' })
    window.location.hash = '#/chat/12'
    await vi.waitFor(() => expect(window.location.hash).toBe('#/chat/12'))
    await new Promise((r) => setTimeout(r, 50))
    expect(screen.getByRole('link', { name: 'AI 對話' })).toBe(link)
    expect(screen.queryByText('載入中…')).not.toBeInTheDocument()
    const meCalls = vi.mocked(fetch).mock.calls.filter((c) => c[0] === '/api/me')
    expect(meCalls).toHaveLength(1)
  })
})
