import { useEffect, useRef, useState } from 'react'
import { ApiError, api } from '../../api/client'
import { readEvents } from '../../api/stream'
import { useNavGuard } from '../../shell/navGuard'
import {
  INTERRUPTED,
  applyEvent,
  canRetryOf,
  failItem,
  fromMessages,
  type ChatItem,
} from './chatState'

export type { ChatItem } from './chatState'

interface Options {
  conversationId: number | null
  onConversationCreated(id: number): void
  onFinished(): void
}

async function loadItems(id: number): Promise<ChatItem[]> {
  const detail = await api.getConversation(id)
  return fromMessages(detail.messages)
}

function loadErrorText(e: unknown): string {
  if (e instanceof ApiError) return e.status === 404 ? '找不到這個對話' : e.message
  return '無法連線到伺服器'
}

function streamErrorText(e: unknown): string {
  if (e instanceof ApiError) return e.message
  if (e instanceof SyntaxError) return INTERRUPTED
  return '無法連線到伺服器'
}

export function useChat({ conversationId, onConversationCreated, onFinished }: Options) {
  const { setBlocked } = useNavGuard()
  const [items, setItems] = useState<ChatItem[]>([])
  const [streaming, setStreaming] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const streamingRef = useRef(false)
  const activeIdRef = useRef(conversationId)
  const createdIdRef = useRef<number | null>(null)
  const counterRef = useRef(0)

  useEffect(() => {
    activeIdRef.current = conversationId
    // The id changed because send() just created this conversation: keep the in-flight items.
    if (conversationId !== null && conversationId === createdIdRef.current) return
    createdIdRef.current = null
    setLoadError(null)
    setItems([])
    if (conversationId === null) return
    let cancelled = false
    loadItems(conversationId).then(
      (loaded) => {
        if (!cancelled) setItems(loaded)
      },
      (e) => {
        if (!cancelled) setLoadError(loadErrorText(e))
      },
    )
    return () => {
      cancelled = true
    }
  }, [conversationId])

  useEffect(() => {
    setBlocked(streaming ? '回覆產生中，確定要離開？' : null)
    return () => setBlocked(null)
  }, [streaming, setBlocked])

  async function run(key: string, open: () => Promise<Response>, reloadOn409 = false) {
    streamingRef.current = true
    setStreaming(true)
    const update = (fn: (i: ChatItem) => ChatItem) =>
      setItems((prev) => prev.map((i) => (i.key === key ? fn(i) : i)))
    let finished = false
    try {
      const res = await open()
      if (!res.body) throw new SyntaxError('empty body')
      for await (const event of readEvents(res.body)) {
        update((i) => applyEvent(i, event, Date.now()))
        if (event.type === 'done') {
          finished = true
          onFinished()
        } else if (event.type === 'error') {
          finished = true
        }
      }
      if (!finished) update((i) => failItem(i, INTERRUPTED, Date.now()))
    } catch (e) {
      if (finished) {
        // already settled by a terminal event
      } else if (reloadOn409 && e instanceof ApiError && e.status === 409) {
        const id = conversationId
        try {
          const loaded = await loadItems(id as number)
          if (activeIdRef.current === id) setItems(loaded)
        } catch (e2) {
          update((i) => failItem(i, streamErrorText(e2), Date.now()))
        }
      } else {
        update((i) => failItem(i, streamErrorText(e), Date.now()))
      }
    } finally {
      streamingRef.current = false
      setStreaming(false)
    }
  }

  function newAssistant(key: string): ChatItem {
    return { key, role: 'assistant', content: '', thinking: '', status: 'streaming' }
  }

  async function send(text: string) {
    if (streamingRef.current) return
    const n = ++counterRef.current
    const assistantKey = `a${n}`
    setItems((prev) => [
      ...prev,
      { key: `u${n}`, role: 'user', content: text, thinking: '', status: 'done' },
      newAssistant(assistantKey),
    ])
    await run(assistantKey, async () => {
      let id = conversationId
      if (id === null) {
        const created = await api.createConversation()
        id = created.id
        createdIdRef.current = id
        onConversationCreated(id)
      }
      return api.sendMessage(id, text)
    })
  }

  async function retry() {
    if (streamingRef.current || !canRetryOf(items, false)) return
    const last = items[items.length - 1]
    const base = last.role === 'assistant' ? items.slice(0, -1) : items
    if (conversationId === null) {
      // the conversation was never created: send the last user message again
      const lastUser = base[base.length - 1]
      setItems(base.slice(0, -1))
      await send(lastUser.content)
      return
    }
    const n = ++counterRef.current
    const assistantKey = `a${n}`
    setItems([...base, newAssistant(assistantKey)])
    await run(assistantKey, () => api.regenerate(conversationId), true)
  }

  return {
    items,
    streaming,
    canRetry: canRetryOf(items, streaming),
    loadError,
    send,
    retry,
  }
}
