import { useEffect, useRef, useState } from 'react'
import { UploadCloud } from 'lucide-react'
import { errorText } from '../../api/client'
import type { ParserInfo } from '../../api/types'
import styles from './UploadPanel.module.css'

export const ACCEPT = '.pdf,.txt,.md,.png,.jpg,.jpeg,.webp'
const EXTENSIONS = ACCEPT.split(',')

// .txt and .md always go through the text parser on the server
export function isTextFile(file: File): boolean {
  return /\.(txt|md)$/i.test(file.name)
}

function isSupported(file: File): boolean {
  const name = file.name.toLowerCase()
  return EXTENSIONS.some((ext) => name.endsWith(ext))
}

type Status = 'waiting' | 'uploading' | 'done' | 'failed'

interface Item {
  id: number
  name: string
  status: Status
  error?: string
}

const STATUS_TEXT: Record<Exclude<Status, 'failed'>, string> = {
  waiting: '等待中',
  uploading: '上傳中…',
  done: '已上傳',
}

interface Props {
  /** The knowledge base's parser for PDFs and images (null while unknown). */
  parser: ParserInfo | null
  /** Sends one file; rejects with the server's message on failure. */
  upload(file: File): Promise<unknown>
  /** Called after each file the server accepted (to refresh the document list). */
  onUploaded(): void
}

/**
 * Drop files on the zone, or click it to pick several: each file uploads right away,
 * one after another, with its own status row. PDFs and images go through the knowledge
 * base's parser (set in the knowledge base settings, not here).
 */
export default function UploadPanel({ parser, upload, onUploaded }: Props) {
  const [items, setItems] = useState<Item[]>([])
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const nextId = useRef(1)
  // uploads run one at a time: each new file waits for the previous ones
  const queue = useRef<Promise<void>>(Promise.resolve())
  const dragDepth = useRef(0)

  // a file dropped next to the zone would otherwise make the browser leave the app
  useEffect(() => {
    const block = (e: DragEvent) => e.preventDefault()
    window.addEventListener('dragover', block)
    window.addEventListener('drop', block)
    return () => {
      window.removeEventListener('dragover', block)
      window.removeEventListener('drop', block)
    }
  }, [])

  const update = (id: number, patch: Partial<Item>) =>
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)))

  const label = parser?.label.split('（')[0] ?? ''

  const start = (files: File[]) => {
    const usable = parser?.available !== false
    for (const file of files) {
      const id = nextId.current++
      let error: string | undefined
      if (!isSupported(file)) error = '不支援的檔案類型（支援 PDF、txt、md、png、jpg、jpeg、webp）'
      else if (!isTextFile(file) && !usable) error = `解析器「${label}」目前無法使用`
      setItems((prev) => [
        ...prev,
        { id, name: file.name, status: error ? 'failed' : 'waiting', error },
      ])
      if (error) continue
      queue.current = queue.current.then(async () => {
        update(id, { status: 'uploading' })
        try {
          await upload(file)
          update(id, { status: 'done' })
          onUploaded()
        } catch (err) {
          update(id, { status: 'failed', error: errorText(err, '上傳失敗') })
        }
      })
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    start(Array.from(e.dataTransfer.files ?? []))
  }

  const busy = items.some((it) => it.status === 'waiting' || it.status === 'uploading')
  const finished = items.some((it) => it.status === 'done' || it.status === 'failed')

  return (
    <div className={styles.panel}>
      {parser && (
        <p className={styles.hint}>
          PDF 與圖片會用「{label}」解析（在上方的知識庫設定中更改）
          {!parser.available && parser.reason && (
            <span className={styles.failed}>
              <br />
              {label}：{parser.reason}
            </span>
          )}
        </p>
      )}

      <div
        role="button"
        tabIndex={0}
        aria-label="拖曳檔案到這裡，或按一下選擇檔案（可多選）"
        className={styles.zone}
        data-dragging={dragging ? 'true' : 'false'}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            input.current?.click()
          }
        }}
        onDragEnter={(e) => {
          e.preventDefault()
          dragDepth.current += 1
          setDragging(true)
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1)
          if (dragDepth.current === 0) setDragging(false)
        }}
        onDrop={onDrop}
      >
        <UploadCloud size={28} aria-hidden />
        <span className={styles.zoneTitle}>拖曳檔案到這裡，或按一下選擇檔案</span>
        <span className={styles.hint}>可一次選多個；支援 PDF、txt、md、png、jpg、jpeg、webp</span>
      </div>
      <input
        ref={input}
        className={styles.hidden}
        type="file"
        accept={ACCEPT}
        multiple
        aria-label="選擇檔案"
        onChange={(e) => {
          start(Array.from(e.target.files ?? []))
          e.target.value = '' // picking the same file again still triggers a change
        }}
      />

      {items.length > 0 && (
        <>
          <ul className={styles.list} aria-label="上傳進度">
            {items.map((it) => (
              <li key={it.id} aria-label={it.name} className={styles.item}>
                <span className={styles.name}>{it.name}</span>
                <span className={it.status === 'failed' ? styles.failed : styles.status}>
                  {it.status === 'failed' ? it.error : STATUS_TEXT[it.status]}
                </span>
              </li>
            ))}
          </ul>
          {finished && !busy && (
            <button
              type="button"
              className={styles.clear}
              onClick={() => setItems((prev) => prev.filter((it) => it.status === 'waiting' || it.status === 'uploading'))}
            >
              清除已完成
            </button>
          )}
        </>
      )}
    </div>
  )
}
