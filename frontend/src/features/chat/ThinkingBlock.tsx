import { useEffect, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import styles from './ThinkingBlock.module.css'

interface Props {
  thinking: string
  startedAt?: number
  seconds?: number
  live: boolean
}

export default function ThinkingBlock({ thinking, startedAt, seconds, live }: Props) {
  const [now, setNow] = useState(() => Date.now())
  const [open, setOpen] = useState<boolean | null>(null)

  useEffect(() => {
    if (!live) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [live])

  const expanded = live ? true : (open ?? false)
  const elapsed = startedAt === undefined ? 0 : Math.max(0, Math.floor((now - startedAt) / 1000))
  const title = live ? `思考中… ${elapsed} 秒` : `已思考 ${seconds ?? 0} 秒`

  return (
    <div className={styles.block}>
      <button
        type="button"
        className={styles.header}
        aria-expanded={expanded}
        onClick={() => setOpen(!expanded)}
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {title}
      </button>
      {expanded && <div className={styles.text}>{thinking}</div>}
    </div>
  )
}
