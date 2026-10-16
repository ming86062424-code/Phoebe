import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

function mockMe(status: number) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify(
          status === 200 ? { id: 1, username: 'amy' } : { detail: '未登入' },
        ),
        { status },
      ),
    ),
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

  it('redirects / to /chat', async () => {
    mockMe(200)
    window.location.hash = '#/'
    render(<App />)
    await screen.findByText('amy')
    expect(window.location.hash).toBe('#/chat')
  })
})
