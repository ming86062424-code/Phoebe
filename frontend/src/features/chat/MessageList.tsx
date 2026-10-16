import { useEffect, useRef } from 'react'
import type { ChatItem } from './chatState'
import MessageBubble from './MessageBubble'
import styles from './MessageList.module.css'

interface Props {
  items: ChatItem[]
  canRetry: boolean
  onRetry(): void
}

export default function MessageList({ items, canRetry, onRetry }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [items, canRetry])

  return (
    <div className={styles.scroll} ref={ref}>
      <div className={styles.inner}>
        {items.length === 0 && <p className={styles.empty}>開始一段新的對話吧</p>}
        {items.map((item) => (
          <MessageBubble key={item.key} item={item} />
        ))}
        {canRetry && (
          <div className={styles.retry}>
            <button type="button" className={styles.retryButton} onClick={onRetry}>
              重新送出
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
