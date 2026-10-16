import { useEffect, useState } from 'react'
import { Plus, Trash2 } from 'lucide-react'
import { api } from '../../api/client'
import type { Conversation } from '../../api/types'
import ConfirmDialog from '../../shell/ConfirmDialog'
import styles from './ConversationList.module.css'

interface Props {
  activeId: number | null
  disabled: boolean
  refreshKey: number
  onSelect(id: number): void
  onNew(): void
  onDeleted(id: number): void
}

export default function ConversationList({
  activeId,
  disabled,
  refreshKey,
  onSelect,
  onNew,
  onDeleted,
}: Props) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [error, setError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<Conversation | null>(null)

  useEffect(() => {
    let cancelled = false
    api.listConversations().then(
      (list) => {
        if (cancelled) return
        setConversations(list)
        setError(null)
      },
      (e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : '無法載入對話列表')
      },
    )
    return () => {
      cancelled = true
    }
  }, [refreshKey])

  const confirmDelete = async () => {
    const target = pendingDelete
    setPendingDelete(null)
    if (!target) return
    try {
      await api.deleteConversation(target.id)
      setConversations((prev) => prev.filter((c) => c.id !== target.id))
      onDeleted(target.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : '刪除失敗')
    }
  }

  return (
    <nav className={styles.list} aria-label="對話列表">
      <button type="button" className={styles.new} disabled={disabled} onClick={onNew}>
        <Plus size={16} />
        新對話
      </button>
      {error && (
        <div role="alert" className={styles.error}>
          {error}
        </div>
      )}
      <ul className={styles.items}>
        {conversations.map((c) => (
          <li key={c.id} className={styles.row}>
            <button
              type="button"
              className={`${styles.item} ${c.id === activeId ? styles.active : ''}`}
              disabled={disabled}
              aria-current={c.id === activeId ? 'page' : undefined}
              onClick={() => onSelect(c.id)}
            >
              {c.title || '新對話'}
            </button>
            <button
              type="button"
              className={styles.delete}
              disabled={disabled}
              aria-label={`刪除對話 ${c.title || '新對話'}`}
              onClick={() => setPendingDelete(c)}
            >
              <Trash2 size={14} />
            </button>
          </li>
        ))}
      </ul>
      {pendingDelete && (
        <ConfirmDialog
          message="確定要刪除這個對話？"
          onConfirm={confirmDelete}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </nav>
  )
}
