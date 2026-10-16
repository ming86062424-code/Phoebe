import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ThinkingBlock from './ThinkingBlock'

afterEach(() => {
  vi.useRealTimers()
})

describe('ThinkingBlock', () => {
  it('ThinkingBlock shows 思考中… while live and 已思考 N 秒 collapsed after', async () => {
    vi.useFakeTimers()
    const start = Date.now()
    const { rerender } = render(<ThinkingBlock thinking="想想" startedAt={start} live />)
    expect(screen.getByRole('button')).toHaveTextContent('思考中…')
    expect(screen.getByText('想想')).toBeVisible()
    act(() => {
      vi.advanceTimersByTime(3000)
    })
    expect(screen.getByRole('button')).toHaveTextContent('3 秒')

    rerender(<ThinkingBlock thinking="想想" startedAt={start} seconds={3} live={false} />)
    expect(screen.getByRole('button')).toHaveTextContent('已思考 3 秒')
    expect(screen.queryByText('想想')).not.toBeInTheDocument()
    vi.useRealTimers()
    await userEvent.click(screen.getByRole('button'))
    expect(screen.getByText('想想')).toBeVisible()
    await userEvent.click(screen.getByRole('button'))
    expect(screen.queryByText('想想')).not.toBeInTheDocument()
  })

  it('clears its timer when unmounted', () => {
    vi.useFakeTimers()
    const { unmount } = render(<ThinkingBlock thinking="a" startedAt={Date.now()} live />)
    expect(vi.getTimerCount()).toBe(1)
    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })
})
