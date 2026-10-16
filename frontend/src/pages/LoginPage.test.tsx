import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HashRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, api } from '../api/client'
import LoginPage from './LoginPage'

vi.mock('../api/client', async (orig) => ({
  ...(await orig<typeof import('../api/client')>()),
  api: { login: vi.fn(), register: vi.fn() },
}))

function setup() {
  return render(
    <HashRouter>
      <LoginPage />
    </HashRouter>,
  )
}

describe('LoginPage', () => {
  beforeEach(() => {
    vi.mocked(api.login).mockReset()
    vi.mocked(api.register).mockReset()
    window.location.hash = '#/login'
  })

  it('logs in and navigates to #/chat', async () => {
    vi.mocked(api.login).mockResolvedValue(undefined)
    setup()
    await userEvent.type(screen.getByLabelText('帳號'), 'amy')
    await userEvent.type(screen.getByLabelText('密碼'), 'secret12')
    await userEvent.click(screen.getByRole('button', { name: '登入' }))
    expect(api.login).toHaveBeenCalledWith('amy', 'secret12')
    await vi.waitFor(() => expect(window.location.hash).toBe('#/chat'))
  })

  it('shows API error message', async () => {
    vi.mocked(api.login).mockRejectedValue(new ApiError(401, '帳號或密碼錯誤'))
    setup()
    await userEvent.type(screen.getByLabelText('帳號'), 'amy')
    await userEvent.type(screen.getByLabelText('密碼'), 'bad')
    await userEvent.click(screen.getByRole('button', { name: '登入' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('帳號或密碼錯誤')
  })

  it('registers then logs in automatically', async () => {
    vi.mocked(api.register).mockResolvedValue(undefined)
    vi.mocked(api.login).mockResolvedValue(undefined)
    setup()
    await userEvent.click(screen.getByRole('button', { name: '還沒有帳號？註冊' }))
    await userEvent.type(screen.getByLabelText('帳號'), 'bob')
    await userEvent.type(screen.getByLabelText('密碼'), 'secret12')
    await userEvent.click(screen.getByRole('button', { name: '註冊' }))
    expect(api.register).toHaveBeenCalledWith('bob', 'secret12')
    await vi.waitFor(() => expect(api.login).toHaveBeenCalledWith('bob', 'secret12'))
    await vi.waitFor(() => expect(window.location.hash).toBe('#/chat'))
  })
})
