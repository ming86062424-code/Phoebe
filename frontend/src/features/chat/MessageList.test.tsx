import { fireEvent, render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ChatItem } from './chatState'
import MessageList from './MessageList'

function item(content: string): ChatItem {
  return { key: 'a', role: 'assistant', content, thinking: '', status: 'streaming' }
}

/** jsdom has no layout: fake the scroll metrics of the list. */
function fakeMetrics(el: HTMLElement, scrollHeight: number, clientHeight: number) {
  Object.defineProperty(el, 'scrollHeight', { configurable: true, value: scrollHeight })
  Object.defineProperty(el, 'clientHeight', { configurable: true, value: clientHeight })
}

describe('MessageList auto-scroll (#7)', () => {
  it('follows the stream only while the user is near the bottom', () => {
    const { container, rerender } = render(
      <MessageList items={[item('一')]} canRetry={false} onRetry={() => {}} />,
    )
    const el = container.firstElementChild as HTMLElement
    fakeMetrics(el, 1000, 400)

    // at the bottom: follows new text
    el.scrollTop = 600
    fireEvent.scroll(el)
    fakeMetrics(el, 1200, 400)
    rerender(<MessageList items={[item('一二')]} canRetry={false} onRetry={() => {}} />)
    expect(el.scrollTop).toBe(1200)

    // scrolled up to read: stays put
    el.scrollTop = 100
    fireEvent.scroll(el)
    fakeMetrics(el, 1400, 400)
    rerender(<MessageList items={[item('一二三')]} canRetry={false} onRetry={() => {}} />)
    expect(el.scrollTop).toBe(100)
  })
})
