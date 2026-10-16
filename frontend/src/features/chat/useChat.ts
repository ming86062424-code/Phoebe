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
  const [loading, setLoading] = useState(conversationId !== null)
  // whether the last user message is stored server-side (so retry may regenerate instead of re-sending)
  const persistedRef = useRef(true)
  const streamingRef = useRef(false)
  const activeIdRef = useRef(conversationId)
  const createdIdRef = useRef<number | null>(null)
  const counterRef = useRef(0)

  useEffect(() => {
    activeIdRef.current = conversationId
    // The id changed because send() just created this conversation: keep the in-flight items.
    if (conversationId !== null && conversationId === createdIdRef.current) return
    createdIdRef.current = null
    persistedRef.current = true
    setLoadError(null)
    setItems([])
    setLoading(conversationId !== null)
    if (conversationId === null) return
    let cancelled = false
    loadItems(conversationId).then(
      (loaded) => {
        if (cancelled) return
        // a reply started meanwhile owns the items: do not overwrite it
        if (!streamingRef.current) setItems(loaded)
        setLoading(false)
      },
      (e) => {
        if (cancelled) return
        setLoadError(loadErrorText(e))
        setLoading(false)
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

  async function run(
    key: string,
    open: () => Promise<Response>,
    { reloadOn409 = false, userKey }: { reloadOn409?: boolean; userKey?: string } = {},
  ): Promise<string | null> {
    streamingRef.current = true
    setStreaming(true)
    const update = (fn: (i: ChatItem) => ChatItem) =>
      setItems((prev) => prev.map((i) => (i.key === key ? fn(i) : i)))
    let finished = false
    let rejected: string | null = null
    try {
      const res = await open()
      if (!res.body) throw new SyntaxError('empty body')
      for await (const event of readEvents(res.body)) {
        update((i) => applyEvent(i, event, Date.now()))
        if (event.type === 'done' || event.type === 'error') {
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
          persistedRef.current = true
          if (activeIdRef.current === id) setItems(loaded)
        } catch (e2) {
          update((i) => failItem(i, streamErrorText(e2), Date.now()))
        }
      } else if (
        userKey !== undefined &&
        e instanceof ApiError &&
        e.status >= 400 &&
        e.status < 500 &&
        e.status !== 401 &&
        !persistedRef.current
      ) {
        // the server refused the message (validation etc.): drop the pair and hand the text back
        rejected = e.message
        setItems((prev) => prev.filter((i) => i.key !== key && i.key !== userKey))
      } else {
        update((i) => failItem(i, streamErrorText(e), Date.now()))
      }
    } finally {
      streamingRef.current = false
      setStreaming(false)
      // the backend may have set the conversation title even when the reply failed
      onFinished()
    }
    return rejected
  }

  function newAssistant(key: string): ChatItem {
    return { key, role: 'assistant', content: '', thinking: '', status: 'streaming' }
  }

  async function send(text: string): Promise<string | null> {
    if (streamingRef.current) return null
    persistedRef.current = false
    const n = ++counterRef.current
    const assistantKey = `a${n}`
    setItems((prev) => [
      ...prev,
      { key: `u${n}`, role: 'user', content: text, thinking: '', status: 'done' },
      newAssistant(assistantKey),
    ])
    return run(
      assistantKey,
      async () => {
        let id = conversationId
        if (id === null) {
          const created = await api.createConversation()
          id = created.id
          createdIdRef.current = id
          onConversationCreated(id)
        }
        try {
          const res = await api.sendMessage(id, text)
          persistedRef.current = true
          return res
        } catch (e) {
          // 503: the backend stores the user message before it calls Ollama
          if (e instanceof ApiError && e.status === 503) persistedRef.current = true
          throw e
        }
      },
      { userKey: `u${n}` },
    )
  }

  async function retry() {
    if (streamingRef.current || !canRetryOf(items, false)) return
    const last = items[items.length - 1]
    const base = last.role === 'assistant' ? items.slice(0, -1) : items
    if (!persistedRef.current) {
      // the message never reached the database: send the same text again
      const lastUser = base[base.length - 1]
      setItems(base.slice(0, -1))
      await send(lastUser.content)
      return
    }
    const n = ++counterRef.current
    const assistantKey = `a${n}`
    setItems([...base, newAssistant(assistantKey)])
    await run(assistantKey, () => api.regenerate(conversationId as number), { reloadOn409: true })
  }

  return {
    items,
    streaming,
    canRetry: canRetryOf(items, streaming),
    loadError,
    loading,
    send,
    retry,
  }
}
