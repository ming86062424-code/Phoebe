import { useEffect, useRef } from 'react'
import type { ChatItem } from './chatState'
import MessageBubble from './MessageBubble'
import styles from './MessageList.module.css'

interface Props {
  items: ChatItem[]
  canRetry: boolean
  onRetry(): void
  kbId?: number | null
}

// within this distance of the bottom the list keeps following new text
const STICK_PX = 80

export default function MessageList({ items, canRetry, onRetry, kbId = null }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const stickRef = useRef(true)

  useEffect(() => {
    const el = ref.current
    if (el && stickRef.current) el.scrollTop = el.scrollHeight
  }, [items, canRetry])

  const onScroll = () => {
    const el = ref.current
    if (el) stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= STICK_PX
  }

  return (
    <div className={styles.scroll} ref={ref} onScroll={onScroll}>
      <div className={styles.inner}>
        {items.length === 0 && <p className={styles.empty}>開始一段新的對話吧</p>}
        {items.map((item) => (
          <MessageBubble key={item.key} item={item} kbId={kbId} />
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
