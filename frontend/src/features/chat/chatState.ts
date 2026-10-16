import type { Message, StreamEvent } from '../../api/types'

export interface ChatItem {
  key: string
  role: 'user' | 'assistant'
  content: string
  thinking: string
  thinkingStartedAt?: number
  thinkingSeconds?: number
  status: 'done' | 'streaming' | 'error'
  error?: string
  interrupted?: boolean
}

export const INTERRUPTED = '回覆中斷'
export const MAX_MESSAGE_LENGTH = 8000

export function fromMessages(messages: Message[]): ChatItem[] {
  return messages.map((m) => ({
    key: `m${m.id}`,
    role: m.role,
    content: m.content,
    thinking: '',
    status: 'done',
  }))
}

function stopThinking(item: ChatItem, now: number): ChatItem {
  if (item.thinkingStartedAt === undefined || item.thinkingSeconds !== undefined) return item
  return {
    ...item,
    thinkingSeconds: Math.floor((now - item.thinkingStartedAt) / 1000),
  }
}

export function failItem(item: ChatItem, detail: string, now: number): ChatItem {
  return {
    ...stopThinking(item, now),
    status: 'error',
    interrupted: item.content !== '',
    error: detail || INTERRUPTED,
  }
}

export function applyEvent(item: ChatItem, event: StreamEvent, now: number): ChatItem {
  switch (event.type) {
    case 'thinking':
      return {
        ...item,
        thinking: item.thinking + event.text,
        thinkingStartedAt: item.thinkingStartedAt ?? now,
      }
    case 'content':
      return { ...stopThinking(item, now), content: item.content + event.text }
    case 'done':
      return { ...stopThinking(item, now), status: 'done' }
    case 'error':
      return failItem(item, event.detail, now)
  }
}

export function canRetryOf(items: ChatItem[], streaming: boolean): boolean {
  if (streaming) return false
  const last = items[items.length - 1]
  if (!last) return false
  return last.role === 'user' || (last.role === 'assistant' && last.status === 'error')
}
