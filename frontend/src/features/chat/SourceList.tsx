import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type { Source } from '../../api/types'
import styles from './SourceList.module.css'

export default function SourceList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)
  if (sources.length === 0) {
    return <div className={styles.none}>知識庫中沒有找到相關內容</div>
  }
  return (
    <div className={styles.block}>
      <button
        type="button"
        className={styles.header}
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        來源（{sources.length}）
      </button>
      {open && (
        <ol className={styles.list}>
          {sources.map((s, i) => (
            <li key={`${s.document_id}-${s.page}-${i}`} className={styles.item}>
              <div className={styles.title}>
                [{i + 1}] {s.filename} 第 {s.page} 頁
              </div>
              <div className={styles.snippet}>{s.text}</div>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
