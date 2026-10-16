import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import Composer from './Composer'

describe('Composer', () => {
  it('Composer clears input immediately on Enter', async () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} onSend={onSend} />)
    const box = screen.getByRole('textbox')
    await userEvent.type(box, '哈囉{Enter}')
    expect(onSend).toHaveBeenCalledWith('哈囉')
    expect(box).toHaveValue('')
  })

  it('Composer ignores whitespace-only input', async () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} onSend={onSend} />)
    const box = screen.getByRole('textbox')
    await userEvent.type(box, '   {Enter}')
    expect(onSend).not.toHaveBeenCalled()
    expect(box).toHaveValue('   ')
  })

  it('Composer does not send while composing (IME)', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} onSend={onSend} />)
    const box = screen.getByRole('textbox')
    fireEvent.change(box, { target: { value: 'ni' } })
    fireEvent.keyDown(box, { key: 'Enter', isComposing: true })
    expect(onSend).not.toHaveBeenCalled()
    expect(box).toHaveValue('ni')
  })

  it('Shift+Enter does not send', async () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} onSend={onSend} />)
    await userEvent.type(screen.getByRole('textbox'), 'a{Shift>}{Enter}{/Shift}b')
    expect(onSend).not.toHaveBeenCalled()
  })

  it('does not send while disabled', async () => {
    const onSend = vi.fn()
    render(<Composer disabled onSend={onSend} />)
    await userEvent.type(screen.getByRole('textbox'), 'a{Enter}')
    expect(onSend).not.toHaveBeenCalled()
  })

  it('blocks messages over 8000 characters with a hint', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} onSend={onSend} />)
    const box = screen.getByRole('textbox')
    fireEvent.change(box, { target: { value: 'a'.repeat(8001) } })
    expect(screen.getByText('訊息最多 8000 字')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '送出' })).toBeDisabled()
    fireEvent.keyDown(box, { key: 'Enter' })
    expect(onSend).not.toHaveBeenCalled()
    fireEvent.change(box, { target: { value: 'a'.repeat(8000) + '   ' } })
    expect(screen.queryByText('訊息最多 8000 字')).toBeNull()
    expect(screen.getByRole('button', { name: '送出' })).toBeEnabled()
  })
})
