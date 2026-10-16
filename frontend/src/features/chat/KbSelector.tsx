import { useEffect, useState } from 'react'
import { BookOpen } from 'lucide-react'
import { api } from '../../api/client'
import type { KnowledgeBase } from '../../api/types'
import styles from './KbSelector.module.css'

interface Props {
  value: number | null
  disabled: boolean
  onChange(id: number | null): void
}

export default function KbSelector({ value, disabled, onChange }: Props) {
  const [kbs, setKbs] = useState<KnowledgeBase[] | null>(null)

  useEffect(() => {
    let cancelled = false
    api.listKbs().then(
      (list) => {
        if (!cancelled) setKbs(Array.isArray(list) ? list : [])
      },
      () => {
        if (!cancelled) setKbs([])
      },
    )
    return () => {
      cancelled = true
    }
  }, [])

  // the selected knowledge base may have been deleted
  const missing = kbs !== null && value !== null && !kbs.some((k) => k.id === value)
  useEffect(() => {
    if (missing) onChange(null)
  }, [missing, onChange])

  if (kbs === null || kbs.length === 0) return null
  return (
    <label className={styles.selector}>
      <BookOpen size={16} aria-hidden="true" />
      <select
        aria-label="知識庫"
        value={value === null || missing ? '' : String(value)}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
      >
        <option value="">不使用知識庫</option>
        {kbs.map((k) => (
          <option key={k.id} value={k.id}>
            {k.name}
          </option>
        ))}
      </select>
    </label>
  )
}
