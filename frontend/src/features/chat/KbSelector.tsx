import { useEffect, useState } from 'react'
import { BookOpen } from 'lucide-react'
import { api } from '../../api/client'
import type { KnowledgeBase } from '../../api/types'
import styles from './KbSelector.module.css'

interface Props {
  value: number | null
  disabled: boolean
  /** problem saving the choice, shown next to the selector */
  error: string | null
  onChange(id: number | null): void
}

export default function KbSelector({ value, disabled, error, onChange }: Props) {
  const [kbs, setKbs] = useState<KnowledgeBase[] | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)

  useEffect(() => {
    let cancelled = false
    api.listKbs().then(
      (list) => {
        if (!cancelled) setKbs(Array.isArray(list) ? list : [])
      },
      () => {
        if (!cancelled) setLoadFailed(true)
      },
    )
    return () => {
      cancelled = true
    }
  }, [])

  // Never change the conversation on our own: a failed load or a deleted knowledge
  // base is only shown, and the user decides what to pick.
  if (loadFailed) {
    return (
      <div role="alert" className={styles.error}>
        無法載入知識庫列表
      </div>
    )
  }
  if (kbs === null || (kbs.length === 0 && value === null)) return null
  const missing = value !== null && !kbs.some((k) => k.id === value)
  return (
    <div className={styles.wrap}>
      <label className={styles.selector}>
        <BookOpen size={16} aria-hidden="true" />
        <select
          aria-label="知識庫"
          value={value === null ? '' : String(value)}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
        >
          <option value="">不使用知識庫</option>
          {missing && <option value={String(value)}>（知識庫不存在）</option>}
          {kbs.map((k) => (
            <option key={k.id} value={k.id}>
              {k.name}
            </option>
          ))}
        </select>
      </label>
      {error && (
        <span role="alert" className={styles.error}>
          {error}
        </span>
      )}
    </div>
  )
}
