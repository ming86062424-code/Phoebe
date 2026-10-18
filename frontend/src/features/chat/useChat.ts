import { useCallback, useEffect, useRef, useState } from 'react'
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
  return (await loadConversation(id)).items
}

async function loadConversation(id: number): Promise<{ items: ChatItem[]; kbId: number | null }> {
  const detail = await api.getConversation(id)
  return { items: fromMessages(detail.messages), kbId: detail.kb_id ?? null }
}

function loadErrorText(e: unknown): string {
  if (e instanceof ApiError) return e.status === 404 ? '找不到這個對話' : e.message
  return '無法連線到伺服器'
}

/** Failures after which we can't tell whether the server stored the user message. */
function isUncertainFailure(e: unknown): boolean {
  if (e instanceof ApiError) return e.status >= 500 && e.status !== 503
  return true // network error: the request may or may not have arrived
}

function abortError(): DOMException {
  return new DOMException('aborted', 'AbortError')
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
  // the last send failed in a way that leaves persistedRef unknown (see isUncertainFailure)
  const uncertainRef = useRef(false)
  const streamingRef = useRef(false)
  const abortRef = useRef<AbortController | null>(null)
  const activeIdRef = useRef(conversationId)
  const createdIdRef = useRef<number | null>(null)
  const counterRef = useRef(0)
  // knowledge base used by the conversation; for a new chat it's applied once the conversation exists
  const [kbId, setKbId] = useState<number | null>(null)
  const kbIdRef = useRef<number | null>(null)
  const applyKbId = useCallback((id: number | null) => {
    kbIdRef.current = id
    setKbId(id)
  }, [])

  /** Stop the running reply; it will not touch state again. */
  const abortRun = useCallback(() => {
    const controller = abortRef.current
    if (!controller) return
    abortRef.current = null
    controller.abort()
    streamingRef.current = false
    setStreaming(false)
  }, [])

  useEffect(() => () => abortRef.current?.abort(), [])

  useEffect(() => {
    activeIdRef.current = conversationId
    // The id changed because send() just created this conversation: keep the in-flight items.
    if (conversationId !== null && conversationId === createdIdRef.current) return
    // a reply still streaming belongs to the conversation we are leaving
    abortRun()
    createdIdRef.current = null
    persistedRef.current = true
    uncertainRef.current = false
    setLoadError(null)
    setItems([])
    setLoading(conversationId !== null)
    if (conversationId === null) return
    let cancelled = false
    loadConversation(conversationId).then(
      (loaded) => {
        if (cancelled) return
        // a reply started meanwhile owns the items: do not overwrite it
        if (!streamingRef.current) setItems(loaded.items)
        applyKbId(loaded.kbId)
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
  }, [conversationId, applyKbId, abortRun])

  useEffect(() => {
    const leave = () => {
      // the user confirmed leaving: drop the reply (the page may stay mounted, e.g. #/chat → #/chat)
      abortRun()
      createdIdRef.current = null
      persistedRef.current = true
      uncertainRef.current = false
      setItems([])
    }
    setBlocked(streaming ? '回覆產生中，確定要離開？' : null, leave)
    return () => setBlocked(null)
  }, [streaming, setBlocked, abortRun])

  async function run(
    key: string,
    open: (signal: AbortSignal) => Promise<Response>,
    { reloadOn409 = false, userKey }: { reloadOn409?: boolean; userKey?: string } = {},
  ): Promise<string | null> {
    const controller = new AbortController()
    const { signal } = controller
    abortRef.current = controller
    streamingRef.current = true
    setStreaming(true)
    const update = (fn: (i: ChatItem) => ChatItem) => {
      if (!signal.aborted) setItems((prev) => prev.map((i) => (i.key === key ? fn(i) : i)))
    }
    let finished = false
    let rejected: string | null = null
    try {
      const res = await open(signal)
      if (!res.body) throw new SyntaxError('empty body')
      for await (const event of readEvents(res.body)) {
        if (signal.aborted) break
        update((i) => applyEvent(i, event, Date.now()))
        if (event.type === 'done' || event.type === 'error') {
          finished = true
        }
      }
      if (!finished) update((i) => failItem(i, INTERRUPTED, Date.now()))
    } catch (e) {
      if (signal.aborted) {
        // left on purpose: nothing to show
      } else if (finished) {
        // already settled by a terminal event
      } else if (reloadOn409 && e instanceof ApiError && e.status === 409) {
        const id = conversationId ?? createdIdRef.current
        try {
          const loaded = await loadItems(id as number)
          persistedRef.current = true
          if (activeIdRef.current === id && !signal.aborted) setItems(loaded)
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
      } else if (!persistedRef.current && isUncertainFailure(e)) {
        uncertainRef.current = true
        update((i) => failItem(i, streamErrorText(e), Date.now()))
      } else {
        update((i) => failItem(i, streamErrorText(e), Date.now()))
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        streamingRef.current = false
        setStreaming(false)
        // the backend may have set the conversation title even when the reply failed
        onFinished()
      }
    }
    return rejected
  }

  function newAssistant(key: string): ChatItem {
    return { key, role: 'assistant', content: '', thinking: '', status: 'streaming' }
  }

  async function send(text: string): Promise<string | null> {
    if (streamingRef.current) return null
    persistedRef.current = false
    uncertainRef.current = false
    const n = ++counterRef.current
    const assistantKey = `a${n}`
    setItems((prev) => [
      ...prev,
      { key: `u${n}`, role: 'user', content: text, thinking: '', status: 'done' },
      newAssistant(assistantKey),
    ])
    return run(
      assistantKey,
      async (signal) => {
        let id = conversationId ?? createdIdRef.current
        if (id === null) {
          const created = await api.createConversation(signal)
          if (signal.aborted) throw abortError()
          id = created.id
          createdIdRef.current = id
          onConversationCreated(id)
          if (kbIdRef.current !== null) await api.setConversationKb(id, kbIdRef.current)
        }
        try {
          const res = await api.sendMessage(id, text, signal)
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
    const lastUser = base[base.length - 1]
    const id = conversationId ?? createdIdRef.current
    if (!persistedRef.current && uncertainRef.current && id !== null) {
      // the failed send may still have stored the message: ask the server
      uncertainRef.current = false
      try {
        const detail = await api.getConversation(id)
        const stored = detail.messages[detail.messages.length - 1]
        if (stored?.role === 'user' && stored.content === lastUser.content) {
          persistedRef.current = true
        }
      } catch {
        // can't tell: sending again is the safer guess
      }
    }
    if (!persistedRef.current || id === null) {
      // the message never reached the database: send the same text again
      setItems(base.slice(0, -1))
      await send(lastUser.content)
      return
    }
    const n = ++counterRef.current
    const assistantKey = `a${n}`
    setItems([...base, newAssistant(assistantKey)])
    await run(assistantKey, (signal) => api.regenerate(id, signal), { reloadOn409: true })
  }

  /** Choose the knowledge base; returns an error message when saving fails. */
  const selectKb = useCallback(
    async (id: number | null): Promise<string | null> => {
      if (id === kbIdRef.current) return null
      if (conversationId === null) {
        applyKbId(id)
        return null
      }
      try {
        const detail = await api.setConversationKb(conversationId, id)
        if (activeIdRef.current === conversationId) applyKbId(detail.kb_id)
        return null
      } catch (e) {
        return e instanceof Error ? e.message : '無法設定知識庫'
      }
    },
    [conversationId, applyKbId],
  )

  return {
    kbId,
    selectKb,
    items,
    streaming,
    canRetry: canRetryOf(items, streaming),
    loadError,
    loading,
    send,
    retry,
  }
}
