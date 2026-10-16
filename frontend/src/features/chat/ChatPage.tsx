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
  const conversationId = parseId(useParams()['*'])
  const navigate = useNavigate()
  const [listCollapsed, setListCollapsed] = usePersistentState('phoebe.chat.listCollapsed', false)
  const narrow = useMediaQuery('(max-width: 1023px)')
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
    if (loadError) navigate('/chat', { replace: true })
  }, [loadError, navigate])

  const collapsed = narrow || listCollapsed
  const go = (path: string) => {
    setNotice(null)
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
            disabled={narrow}
            aria-label={collapsed ? '展開對話列表' : '收合對話列表'}
            onClick={() => setListCollapsed(!listCollapsed)}
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
          disabled={chat.streaming}
          onSend={(text) => {
            setNotice(null)
            void chat.send(text)
          }}
        />
      </section>
    </div>
  )
}
