import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type { Source } from '../../api/types'
import { SourceCards } from '../kb/SourceCard'
import styles from './SourceList.module.css'

interface Props {
  sources: Source[]
  // the conversation's knowledge base, for the image URLs
  kbId: number | null
}

export default function SourceList({ sources, kbId }: Props) {
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
        參考資料（{sources.length}）
      </button>
      {open && <SourceCards sources={sources} kbId={kbId} />}
    </div>
  )
}
