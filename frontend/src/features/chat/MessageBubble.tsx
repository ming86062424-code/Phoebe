import type { ChatItem } from './chatState'
import { INTERRUPTED } from './chatState'
import ThinkingBlock from './ThinkingBlock'
import styles from './MessageBubble.module.css'

export default function MessageBubble({ item }: { item: ChatItem }) {
  const isUser = item.role === 'user'
  const live = item.status === 'streaming' && item.thinkingSeconds === undefined
  const showAlert = item.status === 'error' && !(item.interrupted && item.error === INTERRUPTED)
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
        {!isUser && item.status === 'streaming' && item.content === '' && item.thinking === '' && (
          <span className={styles.pending}>…</span>
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
