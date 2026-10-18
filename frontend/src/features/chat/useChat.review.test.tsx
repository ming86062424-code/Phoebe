import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { NavGuardProvider, useNavGuard } from '../../shell/navGuard'
import { conversation, heldResponse, jsonResponse, ndjsonResponse } from './testUtils'
import { useChat } from './useChat'

const wrapper = ({ children }: { children: ReactNode }) => (
  <NavGuardProvider>{children}</NavGuardProvider>
)

const fetchMock = vi.fn()

function setup(conversationId: number | null = 1) {
  const onConversationCreated = vi.fn()
  const onFinished = vi.fn()
  const hook = renderHook(
    (props: { id: number | null }) => ({
      chat: useChat({ conversationId: props.id, onConversationCreated, onFinished }),
      guard: useNavGuard(),
    }),
    { wrapper, initialProps: { id: conversationId } },
  )
  return { ...hook, onConversationCreated, onFinished }
}

/** A fetch response that never resolves until its signal aborts. */
function pendingUntilAborted(init?: RequestInit): Promise<Response> {
  return new Promise((_, reject) => {
    init?.signal?.addEventListener('abort', () =>
      reject(new DOMException('aborted', 'AbortError')),
    )
  })
}

function urls(): string[] {
  return fetchMock.mock.calls.map((c) => String(c[0]))
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useChat review fixes', () => {
  it('#1 loads the other conversation when switching away mid-stream', async () => {
    const held = heldResponse()
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/conversations/5')
        return Promise.resolve(jsonResponse(conversation(5, [{ role: 'user', content: '五' }])))
      if (url === '/api/conversations/5/messages') return Promise.resolve(held.response)
      if (url === '/api/conversations/3')
        return Promise.resolve(jsonResponse(conversation(3, [{ role: 'user', content: '三' }])))
      return Promise.reject(new Error('unexpected ' + url))
    })
    const { result, rerender } = setup(5)
    await waitFor(() => expect(result.current.chat.loading).toBe(false))
    act(() => {
      void result.current.chat.send('還在串流')
    })
    await waitFor(() => expect(urls()).toContain('/api/conversations/5/messages'))
    rerender({ id: 3 })
    await waitFor(() => expect(result.current.chat.items.map((i) => i.content)).toEqual(['三']))
    expect(result.current.chat.streaming).toBe(false)
  })

  it('#2 aborts the stream when switching conversations and ignores it afterwards', async () => {
    const signals: AbortSignal[] = []
    const held = heldResponse()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/conversations/5') return Promise.resolve(jsonResponse(conversation(5)))
      if (url === '/api/conversations/5/messages') {
        if (init?.signal) signals.push(init.signal)
        return Promise.resolve(held.response)
      }
      if (url === '/api/conversations/3') return Promise.resolve(jsonResponse(conversation(3)))
      return Promise.reject(new Error('unexpected ' + url))
    })
    const { result, rerender, onFinished } = setup(5)
    await waitFor(() => expect(result.current.chat.loading).toBe(false))
    act(() => {
      void result.current.chat.send('x')
    })
    await waitFor(() => expect(signals).toHaveLength(1))
    rerender({ id: 3 })
    await waitFor(() => expect(signals[0].aborted).toBe(true))
    expect(result.current.chat.streaming).toBe(false)
    expect(onFinished).not.toHaveBeenCalled()
  })

  it('#2 confirming leave while a new conversation is being created never navigates to it', async () => {
    let resolveCreate!: (r: Response) => void
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/conversations' && init?.method === 'POST')
        return new Promise<Response>((r) => (resolveCreate = r))
      if (url === '/api/conversations/9/messages') return pendingUntilAborted(init)
      return Promise.reject(new Error('unexpected ' + url))
    })
    const { result, onConversationCreated } = setup(null)
    act(() => {
      void result.current.chat.send('x')
    })
    let left!: Promise<boolean>
    act(() => {
      left = result.current.guard.confirmLeave()
    })
    // the confirm dialog is rendered by the provider; answer it
    const confirm = await waitFor(() => {
      const button = document.querySelector<HTMLButtonElement>('[role=alertdialog] button:last-child')
      expect(button).not.toBeNull()
      return button!
    })
    act(() => confirm.click())
    expect(await left).toBe(true)
    expect(result.current.chat.streaming).toBe(false)
    expect(result.current.chat.items).toEqual([])
    await act(async () => {
      resolveCreate(jsonResponse({ ...conversation(9), messages: undefined }, 201))
    })
    expect(onConversationCreated).not.toHaveBeenCalled()
    expect(urls()).not.toContain('/api/conversations/9/messages')
  })

  it('#2 aborts the stream on unmount', async () => {
    const signals: AbortSignal[] = []
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/conversations/1') return Promise.resolve(jsonResponse(conversation(1)))
      if (url === '/api/conversations/1/messages') {
        if (init?.signal) signals.push(init.signal)
        return pendingUntilAborted(init)
      }
      return Promise.reject(new Error('unexpected ' + url))
    })
    const { result, unmount } = setup(1)
    await waitFor(() => expect(result.current.chat.loading).toBe(false))
    act(() => {
      void result.current.chat.send('x')
    })
    await waitFor(() => expect(signals).toHaveLength(1))
    unmount()
    expect(signals[0].aborted).toBe(true)
  })

  it('#6 retry never calls /api/conversations/null/regenerate', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ ...conversation(7), messages: undefined }, 201))
      .mockResolvedValueOnce(jsonResponse({ detail: '無法連線到 Ollama' }, 503))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'content', text: '好' },
          { type: 'done', message_id: 9 },
        ]),
      )
    // the parent never re-renders with the new id
    const { result } = setup(null)
    await act(() => result.current.chat.send('x'))
    await act(() => result.current.chat.retry())
    expect(urls()).not.toContain('/api/conversations/null/regenerate')
    expect(urls()[2]).toBe('/api/conversations/7/regenerate')
    expect(result.current.chat.items.map((i) => i.content)).toEqual(['x', '好'])
  })

  it('#10 after a 500, retry regenerates when the server already stored the message', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(jsonResponse({ detail: 'Internal Server Error' }, 500))
      .mockResolvedValueOnce(jsonResponse(conversation(1, [{ role: 'user', content: 'x' }])))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'content', text: '好' },
          { type: 'done', message_id: 9 },
        ]),
      )
    const { result } = setup(1)
    await waitFor(() => expect(result.current.chat.loading).toBe(false))
    await act(() => result.current.chat.send('x'))
    await act(() => result.current.chat.retry())
    expect(urls().slice(2)).toEqual(['/api/conversations/1', '/api/conversations/1/regenerate'])
    expect(result.current.chat.items.map((i) => i.content)).toEqual(['x', '好'])
  })

  it('#10 after a network error, retry re-sends when the message is not stored', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'content', text: '好' },
          { type: 'done', message_id: 9 },
        ]),
      )
    const { result } = setup(1)
    await waitFor(() => expect(result.current.chat.loading).toBe(false))
    await act(() => result.current.chat.send('x'))
    await act(() => result.current.chat.retry())
    expect(urls().slice(2)).toEqual(['/api/conversations/1', '/api/conversations/1/messages'])
  })
})
