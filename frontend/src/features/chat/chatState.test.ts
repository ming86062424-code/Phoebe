import { describe, expect, it } from 'vitest'
import { applyEvent, canRetryOf, failItem, type ChatItem } from './chatState'

const fresh: ChatItem = {
  key: 'a',
  role: 'assistant',
  content: '',
  thinking: '',
  status: 'streaming',
}

describe('chatState', () => {
  it('records thinking start once and appends text', () => {
    let item = applyEvent(fresh, { type: 'thinking', text: '甲' }, 1000)
    item = applyEvent(item, { type: 'thinking', text: '乙' }, 2000)
    expect(item).toMatchObject({ thinking: '甲乙', thinkingStartedAt: 1000 })
  })

  it('rounds thinking seconds at the first content only', () => {
    let item = applyEvent(fresh, { type: 'thinking', text: '甲' }, 1000)
    item = applyEvent(item, { type: 'content', text: 'x' }, 3600)
    item = applyEvent(item, { type: 'content', text: 'y' }, 9000)
    expect(item).toMatchObject({ content: 'xy', thinkingSeconds: 3 })
  })

  it('leaves thinkingSeconds unset when there was no thinking', () => {
    const item = applyEvent(fresh, { type: 'content', text: 'x' }, 5000)
    expect(item.thinkingSeconds).toBeUndefined()
  })

  it('failItem uses the fallback detail and sets interrupted from content', () => {
    expect(failItem(fresh, '', 0)).toMatchObject({ error: '回覆中斷', interrupted: false })
    expect(failItem({ ...fresh, content: 'a' }, 'boom', 0)).toMatchObject({
      error: 'boom',
      interrupted: true,
    })
  })

  it('canRetryOf', () => {
    const user: ChatItem = { ...fresh, role: 'user', status: 'done' }
    expect(canRetryOf([], false)).toBe(false)
    expect(canRetryOf([user], false)).toBe(true)
    expect(canRetryOf([user], true)).toBe(false)
    expect(canRetryOf([user, { ...fresh, status: 'error' }], false)).toBe(true)
    expect(canRetryOf([user, { ...fresh, status: 'done' }], false)).toBe(false)
  })
})
