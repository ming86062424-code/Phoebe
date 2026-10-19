import type { ChatItem } from './chatState'
import { INTERRUPTED } from './chatState'
import SourceList from './SourceList'
import ThinkingBlock from './ThinkingBlock'
import styles from './MessageBubble.module.css'

interface Props {
  item: ChatItem
  // the conversation's knowledge base, for source images
  kbId?: number | null
}

export default function MessageBubble({ item, kbId = null }: Props) {
  const isUser = item.role === 'user'
  const live = item.status === 'streaming' && item.thinkingSeconds === undefined
  const showAlert = item.status === 'error' && !(item.interrupted && item.error === INTERRUPTED)
  const isEmptyReply = !isUser && item.status === 'done' && item.content === ''
  return (
    <div className={`${styles.row} ${isUser ? styles.user : styles.assistant}`}>
      <div className={styles.bubble}>
        {!isUser && item.thinking !== '' && (
          <ThinkingBlock
            thinking={item.thinking}
            startedAt={item.thinkingStartedAt}
            seconds={item.thinkingSeconds}
            live={live}
          />
        )}
        {(item.content !== '' || item.interrupted) && (
          <div className={styles.content}>
            {item.content}
            {item.interrupted ? '（回覆中斷）' : ''}
          </div>
        )}
        {isEmptyReply && <div className={styles.empty}>（沒有回答內容）</div>}
        {!isUser && item.status === 'streaming' && item.content === '' && item.thinking === '' && (
          <span className={styles.pending}>…</span>
        )}
        {!isUser && item.sources && item.status !== 'streaming' && (
          <SourceList sources={item.sources} kbId={kbId} />
        )}
        {showAlert && (
          <div role="alert" className={styles.error}>
            {item.error}
          </div>
        )}
      </div>
    </div>
  )
}
