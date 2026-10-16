import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { NavGuardProvider } from '../../shell/navGuard'
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
    (props: { id: number | null }) =>
      useChat({ conversationId: props.id, onConversationCreated, onFinished }),
    { wrapper, initialProps: { id: conversationId } },
  )
  return { ...hook, onConversationCreated, onFinished }
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useChat', () => {
  it('streams thinking then content then done', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'thinking', text: '嗯' },
          { type: 'content', text: '你' },
          { type: 'content', text: '好' },
          { type: 'done', message_id: 5 },
        ]),
      )
    const { result, onFinished } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('哈囉'))
    expect(result.current.items).toHaveLength(2)
    expect(result.current.items[0]).toMatchObject({ role: 'user', content: '哈囉' })
    expect(result.current.items[1]).toMatchObject({
      role: 'assistant',
      thinking: '嗯',
      content: '你好',
      status: 'done',
    })
    expect(typeof result.current.items[1].thinkingSeconds).toBe('number')
    expect(result.current.streaming).toBe(false)
    expect(onFinished).toHaveBeenCalledTimes(1)
  })

  it('useChat finishes when done arrives without content', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'thinking', text: '想' },
          { type: 'done', message_id: 5 },
        ]),
      )
    const { result } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('x'))
    expect(result.current.items[1].status).toBe('done')
    expect(typeof result.current.items[1].thinkingSeconds).toBe('number')
    expect(result.current.streaming).toBe(false)
  })

  it('marks error event as interrupted and allows retry', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'content', text: '你' },
          { type: 'error', detail: '模型壞了' },
        ]),
      )
    const { result } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('x'))
    expect(result.current.items[1]).toMatchObject({
      status: 'error',
      interrupted: true,
      error: '模型壞了',
      content: '你',
    })
    expect(result.current.canRetry).toBe(true)
  })

  it('treats a stream that ends without done as an error', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(ndjsonResponse([{ type: 'content', text: '你' }]))
    const { result } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('x'))
    expect(result.current.items[1]).toMatchObject({
      status: 'error',
      error: '回覆中斷',
      interrupted: true,
    })
    expect(result.current.canRetry).toBe(true)
  })

  it('treats a malformed NDJSON line as an interrupted reply', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(ndjsonResponse([{ type: 'content', text: '你' }], '{oops\n'))
    const { result } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('x'))
    expect(result.current.items[1]).toMatchObject({ status: 'error', error: '回覆中斷' })
  })

  it('shows 503 detail and allows retry', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(jsonResponse({ detail: '無法連線到 Ollama' }, 503))
    const { result } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('x'))
    expect(result.current.items[1]).toMatchObject({
      status: 'error',
      error: '無法連線到 Ollama',
      interrupted: false,
    })
    expect(result.current.canRetry).toBe(true)
  })

  it('shows a connection message on network failure', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
    const { result } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('x'))
    expect(result.current.items[1].error).toBe('無法連線到伺服器')
  })

  it('retry calls regenerate without adding a user item', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(jsonResponse({ detail: '壞了' }, 503))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'content', text: '好' },
          { type: 'done', message_id: 9 },
        ]),
      )
    const { result } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('x'))
    await act(() => result.current.retry())
    expect(fetchMock.mock.calls[2][0]).toBe('/api/conversations/1/regenerate')
    expect(result.current.items).toHaveLength(2)
    expect(result.current.items[1]).toMatchObject({ content: '好', status: 'done' })
    expect(result.current.canRetry).toBe(false)
  })

  it('reloads the conversation when regenerate answers 409', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1, [{ role: 'user', content: 'q' }])))
      .mockResolvedValueOnce(jsonResponse({ detail: '沒有需要重新產生的回答' }, 409))
      .mockResolvedValueOnce(
        jsonResponse(
          conversation(1, [
            { role: 'user', content: 'q' },
            { role: 'assistant', content: 'a' },
          ]),
        ),
      )
    const { result } = setup()
    await waitFor(() => expect(result.current.canRetry).toBe(true))
    await act(() => result.current.retry())
    expect(result.current.items.map((i) => i.content)).toEqual(['q', 'a'])
    expect(result.current.canRetry).toBe(false)
    expect(result.current.streaming).toBe(false)
  })

  it('canRetry is true when a loaded conversation ends with a user message', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(conversation(1, [{ role: 'user', content: 'q' }])))
    const { result } = setup()
    await waitFor(() => expect(result.current.items).toHaveLength(1))
    expect(result.current.canRetry).toBe(true)
  })

  it('creates a conversation when sending with no conversation selected', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(7), 201))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'content', text: '嗨' },
          { type: 'done', message_id: 1 },
        ]),
      )
    const { result, rerender, onConversationCreated } = setup(null)
    await act(() => result.current.send('哈囉'))
    expect(onConversationCreated).toHaveBeenCalledWith(7)
    expect(fetchMock.mock.calls[1][0]).toBe('/api/conversations/7/messages')
    // parent navigates to the new id: in-flight items must not be reloaded or clobbered
    rerender({ id: 7 })
    await act(async () => {})
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(result.current.items.map((i) => i.content)).toEqual(['哈囉', '嗨'])
  })

  it('sets loadError for a missing conversation', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'x' }, 404))
    const { result } = setup(99)
    await waitFor(() => expect(result.current.loadError).toBe('找不到這個對話'))
  })

  it('discards a late load response after the id changed', async () => {
    let resolveFirst!: (r: Response) => void
    fetchMock
      .mockReturnValueOnce(new Promise<Response>((r) => (resolveFirst = r)))
      .mockResolvedValueOnce(jsonResponse(conversation(2, [{ role: 'assistant', content: 'two' }])))
    const { result, rerender } = setup(1)
    rerender({ id: 2 })
    await waitFor(() => expect(result.current.items).toHaveLength(1))
    await act(async () => {
      resolveFirst(jsonResponse(conversation(1, [{ role: 'assistant', content: 'one' }])))
    })
    expect(result.current.items[0].content).toBe('two')
  })

  it('is streaming until the stream finishes', async () => {
    const held = heldResponse()
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(held.response)
    const { result } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    let p!: Promise<string | null>
    act(() => {
      p = result.current.send('x')
    })
    expect(result.current.streaming).toBe(true)
    expect(result.current.canRetry).toBe(false)
    await act(async () => {
      held.push({ type: 'done', message_id: 1 })
      held.close()
      await p
    })
    expect(result.current.streaming).toBe(false)
  })

  it('keeps a message sent while the conversation is still loading', async () => {
    let resolveLoad!: (r: Response) => void
    const held = heldResponse()
    fetchMock
      .mockReturnValueOnce(new Promise<Response>((r) => (resolveLoad = r)))
      .mockResolvedValueOnce(held.response)
    const { result } = setup()
    expect(result.current.loading).toBe(true)
    let p!: Promise<string | null>
    act(() => {
      p = result.current.send('哈囉')
    })
    await act(async () => {
      resolveLoad(jsonResponse(conversation(1)))
    })
    expect(result.current.items.map((i) => i.content)).toEqual(['哈囉', ''])
    expect(result.current.loading).toBe(false)
    await act(async () => {
      held.push({ type: 'content', text: '嗨' })
      held.push({ type: 'done', message_id: 1 })
      held.close()
      await p
    })
    expect(result.current.items.map((i) => i.content)).toEqual(['哈囉', '嗨'])
  })

  it('a 422 rejection removes the pair, keeps the text and never regenerates', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(
        jsonResponse({ detail: [{ msg: 'Value error, 訊息最多 8000 字' }] }, 422),
      )
    const { result, onFinished } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    let rejected: string | null = null
    await act(async () => {
      rejected = await result.current.send('太長')
    })
    expect(rejected).toBe('訊息最多 8000 字')
    expect(result.current.items).toHaveLength(0)
    expect(result.current.canRetry).toBe(false)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(onFinished).toHaveBeenCalled()
  })

  it('retry after a network error before the message was stored sends it again', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'content', text: '好' },
          { type: 'done', message_id: 9 },
        ]),
      )
    const { result } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('x'))
    expect(result.current.canRetry).toBe(true)
    await act(() => result.current.retry())
    expect(fetchMock.mock.calls[2][0]).toBe('/api/conversations/1/messages')
    expect(JSON.parse(String(fetchMock.mock.calls[2][1].body))).toEqual({ content: 'x' })
    expect(result.current.items.map((i) => i.content)).toEqual(['x', '好'])
  })

  it('retry after createConversation succeeded but sendMessage failed re-sends to the new id', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(7), 201))
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(
        ndjsonResponse([
          { type: 'content', text: '好' },
          { type: 'done', message_id: 9 },
        ]),
      )
    const { result, rerender } = setup(null)
    await act(() => result.current.send('x'))
    rerender({ id: 7 })
    await act(async () => {})
    await act(() => result.current.retry())
    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(fetchMock.mock.calls[2][0]).toBe('/api/conversations/7/messages')
    expect(result.current.items.map((i) => i.content)).toEqual(['x', '好'])
  })

  it('refreshes the list when the stream ends with an error event', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(conversation(1)))
      .mockResolvedValueOnce(ndjsonResponse([{ type: 'error', detail: '壞了' }]))
    const { result, onFinished } = setup()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await act(() => result.current.send('x'))
    expect(onFinished).toHaveBeenCalledTimes(1)
  })
})
