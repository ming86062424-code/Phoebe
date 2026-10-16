import { useEffect, useState } from 'react'
import { PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { useNavigate, useParams } from 'react-router'
import { useMediaQuery } from '../../shell/useMediaQuery'
import { usePersistentState } from '../../shell/usePersistentState'
import Composer from './Composer'
import ConversationList from './ConversationList'
import MessageList from './MessageList'
import { useChat } from './useChat'
import styles from './ChatPage.module.css'

function parseId(raw: string | undefined): number | null {
  return raw && /^\d+$/.test(raw) ? Number(raw) : null
}

export default function ChatPage() {
  const rawId = useParams()['*']
  const conversationId = parseId(rawId)
  const invalidId = rawId !== undefined && rawId !== '' && conversationId === null
  const navigate = useNavigate()
  const [listCollapsed, setListCollapsed] = usePersistentState('phoebe.chat.listCollapsed', false)
  const narrow = useMediaQuery('(max-width: 1023px)')
  // narrow windows start collapsed; opening is temporary and never saved
  const [narrowOpen, setNarrowOpen] = useState(false)
  const [prevNarrow, setPrevNarrow] = useState(narrow)
  if (narrow !== prevNarrow) {
    setPrevNarrow(narrow)
    setNarrowOpen(false)
  }
  const [draft, setDraft] = useState({ text: '', n: 0 })
  const [sendError, setSendError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [notice, setNotice] = useState<string | null>(null)
  const refresh = () => setRefreshKey((k) => k + 1)

  const chat = useChat({
    conversationId,
    onConversationCreated: (id) => {
      refresh()
      navigate(`/chat/${id}`, { replace: true })
    },
    onFinished: refresh,
  })

  const { loadError } = chat
  const [shownError, setShownError] = useState<string | null>(null)
  if (loadError !== shownError && (loadError || shownError)) {
    // keep the message visible after useChat clears it on the redirect to #/chat
    setShownError(loadError)
    if (loadError) setNotice(loadError)
  }
  useEffect(() => {
    if (loadError || invalidId) navigate('/chat', { replace: true })
  }, [loadError, invalidId, navigate])

  const collapsed = narrow ? !narrowOpen : listCollapsed
  const go = (path: string) => {
    setNotice(null)
    if (narrow) setNarrowOpen(false)
    navigate(path)
  }

  return (
    <div className={styles.page}>
      {!collapsed && (
        <ConversationList
          activeId={conversationId}
          disabled={chat.streaming}
          refreshKey={refreshKey}
          onSelect={(id) => go(`/chat/${id}`)}
          onNew={() => go('/chat')}
          onDeleted={(id) => {
            if (id === conversationId) go('/chat')
          }}
        />
      )}
      <section className={styles.chat}>
        <div className={styles.toolbar}>
          <button
            type="button"
            className={styles.toggle}
            aria-label={collapsed ? '展開對話列表' : '收合對話列表'}
            onClick={() => (narrow ? setNarrowOpen(!narrowOpen) : setListCollapsed(!listCollapsed))}
          >
            {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>
        </div>
        {notice && (
          <div role="alert" className={styles.notice}>
            {notice}
          </div>
        )}
        <MessageList items={chat.items} canRetry={chat.canRetry} onRetry={chat.retry} />
        <Composer
          disabled={chat.streaming || chat.loading}
          restore={draft}
          error={sendError}
          onSend={(text) => {
            setNotice(null)
            setSendError(null)
            void chat.send(text).then((rejected) => {
              if (rejected === null) return
              setSendError(rejected)
              setDraft((d) => ({ text, n: d.n + 1 }))
            })
          }}
        />
      </section>
    </div>
  )
}
