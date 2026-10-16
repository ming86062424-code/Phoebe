import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useEffect } from 'react'
import { HashRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { features } from '../features/registry'
import { CurrentUserProvider } from './currentUser'
import { NavGuardProvider, useNavGuard } from './navGuard'
import NavSidebar from './NavSidebar'

vi.mock('../api/client', async (orig) => ({
  ...(await orig<typeof import('../api/client')>()),
  api: { logout: vi.fn().mockResolvedValue(undefined) },
}))

function Blocker({ reason }: { reason: string }) {
  const { setBlocked } = useNavGuard()
  useEffect(() => {
    setBlocked(reason)
  }, [reason, setBlocked])
  return null
}

function setup(reason?: string) {
  return render(
    <HashRouter>
      <CurrentUserProvider user={{ id: 1, username: 'amy' }}>
        <NavGuardProvider>
          {reason && <Blocker reason={reason} />}
          <NavSidebar />
        </NavGuardProvider>
      </CurrentUserProvider>
    </HashRouter>,
  )
}

describe('NavSidebar', () => {
  beforeEach(() => {
    localStorage.clear()
    window.location.hash = '#/start'
  })

  it('renders one link per registry feature', () => {
    setup()
    expect(features).toHaveLength(2)
    expect(screen.getByRole('link', { name: 'AI 對話' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '知識庫' })).toBeInTheDocument()
    expect(screen.getByText('amy')).toBeInTheDocument()
  })

  it('collapse hides labels and persists', async () => {
    setup()
    expect(screen.getByText('AI 對話')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '收合選單' }))
    expect(screen.queryByText('AI 對話')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'AI 對話' })).toHaveAttribute(
      'title',
      'AI 對話',
    )
    expect(localStorage.getItem('phoebe.nav.collapsed')).toBe('true')
  })

  it('navigates when not blocked', async () => {
    setup()
    await userEvent.click(screen.getByRole('link', { name: 'AI 對話' }))
    expect(window.location.hash).toBe('#/chat')
  })

  it('asks before leaving when blocked', async () => {
    setup('回覆產生中，確定要離開？')
    await userEvent.click(screen.getByRole('link', { name: 'AI 對話' }))
    expect(await screen.findByText('回覆產生中，確定要離開？')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '取消' }))
    expect(screen.queryByText('回覆產生中，確定要離開？')).not.toBeInTheDocument()
    expect(window.location.hash).toBe('#/start')
  })

  it('leaves after confirming', async () => {
    setup('回覆產生中，確定要離開？')
    await userEvent.click(screen.getByRole('link', { name: 'AI 對話' }))
    await userEvent.click(await screen.findByRole('button', { name: '確定' }))
    expect(window.location.hash).toBe('#/chat')
  })

  it('logs out and goes to login', async () => {
    setup()
    await userEvent.click(screen.getByRole('button', { name: '登出' }))
    expect(api.logout).toHaveBeenCalled()
    expect(window.location.hash).toBe('#/login')
  })
})
