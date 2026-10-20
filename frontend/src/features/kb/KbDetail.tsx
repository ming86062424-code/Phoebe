import { useCallback, useEffect, useRef, useState } from 'react'
import { Pencil, RotateCcw, Search, Trash2, Upload } from 'lucide-react'
import { api, errorText } from '../../api/client'
import type { KbDocument, KnowledgeBase, ParserInfo, Source } from '../../api/types'
import ConfirmDialog from '../../shell/ConfirmDialog'
import { SourceCards } from './SourceCard'
import styles from './KbDetail.module.css'

const POLL_MS = 2000
const ACCEPT = '.pdf,.txt,.md,.png,.jpg,.jpeg,.webp'

// .txt and .md always go through the text parser on the server
function isTextFile(file: File): boolean {
  return /\.(txt|md)$/i.test(file.name)
}

const STATUS_TEXT: Record<KbDocument['status'], string> = {
  pending: '排隊中',
  queued: '排隊中',
  parsing: '解析中',
  indexing: '建立索引中',
  ready: '完成',
  failed: '失敗',
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function statusText(doc: KbDocument): string {
  if (doc.status === 'parsing' && doc.progress_total > 0) {
    return `解析中 ${doc.progress_done}/${doc.progress_total} 頁`
  }
  return STATUS_TEXT[doc.status]
}

interface Props {
  kb: KnowledgeBase
  onChanged(kb: KnowledgeBase): void
  onDocumentsChanged(): void
}

export default function KbDetail({ kb, onChanged, onDocumentsChanged }: Props) {
  const [docs, setDocs] = useState<KbDocument[] | null>(null)
  const [parsers, setParsers] = useState<ParserInfo[]>([])
  const [parser, setParser] = useState('liteparse')
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [renaming, setRenaming] = useState(false)
  const [name, setName] = useState(kb.name)
  const [pendingDelete, setPendingDelete] = useState<KbDocument | null>(null)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<Source[] | null>(null)
  const [searching, setSearching] = useState(false)
  // hybrid weight for the search test: starts at the chat's default (null until loaded;
  // without it the server's default is used)
  const [defaultWeight, setDefaultWeight] = useState<number | null>(null)
  const [weight, setWeight] = useState<number | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const loadDocs = useCallback(() => {
    return api.listDocuments(kb.id).then(setDocs, (e) => setError(errorText(e, '無法載入文件')))
  }, [kb.id])

  useEffect(() => {
    void loadDocs()
    api.listParsers().then(setParsers, () => setParsers([]))
  }, [loadDocs])

  useEffect(() => {
    api.searchSettings().then(
      (s) => {
        setDefaultWeight(s.vector_weight)
        setWeight(s.vector_weight)
      },
      () => undefined,
    )
  }, [])

  const busy = docs?.some((d) => d.status !== 'ready' && d.status !== 'failed') ?? false
  useEffect(() => {
    if (!busy) return
    const timer = setInterval(() => void loadDocs(), POLL_MS)
    return () => clearInterval(timer)
  }, [busy, loadDocs])

  // refresh the document counts in the list once processing settles
  const prevBusy = useRef(busy)
  useEffect(() => {
    if (prevBusy.current && !busy) onDocumentsChanged()
    prevBusy.current = busy
  }, [busy, onDocumentsChanged])

  const upload = async (e: React.FormEvent) => {
    e.preventDefault()
    if (files.length === 0 || uploading) return
    setUploading(true)
    setError(null)
    const failures: string[] = []
    for (const file of files) {
      try {
        await api.uploadDocument(kb.id, file, parser)
      } catch (err) {
        failures.push(`${file.name}：${errorText(err, '上傳失敗')}`)
      }
    }
    setUploading(false)
    setFiles([])
    if (fileInput.current) fileInput.current.value = ''
    if (failures.length > 0) setError(failures.join('\n'))
    await loadDocs()
    onDocumentsChanged()
  }

  const rename = async (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return
    try {
      onChanged(await api.updateKb(kb.id, { name: trimmed }))
      setRenaming(false)
    } catch (err) {
      setError(errorText(err, '改名失敗'))
    }
  }

  const retry = async (doc: KbDocument) => {
    try {
      await api.retryDocument(kb.id, doc.id)
      await loadDocs()
    } catch (err) {
      setError(errorText(err, '重試失敗'))
    }
  }

  const confirmDelete = async () => {
    const target = pendingDelete
    setPendingDelete(null)
    if (!target) return
    try {
      await api.deleteDocument(kb.id, target.id)
      setDocs((prev) => (prev ?? []).filter((d) => d.id !== target.id))
      onDocumentsChanged()
    } catch (err) {
      setError(errorText(err, '刪除失敗'))
    }
  }

  const search = async (e: React.FormEvent) => {
    e.preventDefault()
    const q = query.trim()
    if (!q || searching) return
    setSearching(true)
    try {
      setResults(await api.search(kb.id, q, weight ?? undefined))
    } catch (err) {
      setError(errorText(err, '檢索失敗'))
    } finally {
      setSearching(false)
    }
  }

  const parserLabel = (id: string) =>
    id === 'text' ? '文字' : (parsers.find((p) => p.id === id)?.label.split('（')[0] ?? id)
  const selectedParser = parsers.find((p) => p.id === parser)
  const needsParser = files.length === 0 || !files.every(isTextFile)

  return (
    <div className={styles.detail}>
      <header className={styles.header}>
        {renaming ? (
          <form className={styles.renameForm} onSubmit={rename}>
            <input
              aria-label="知識庫名稱"
              value={name}
              maxLength={50}
              autoFocus
              onChange={(e) => setName(e.target.value)}
            />
            <button type="submit" className={styles.primary}>
              儲存
            </button>
            <button
              type="button"
              className={styles.secondary}
              onClick={() => {
                setName(kb.name)
                setRenaming(false)
              }}
            >
              取消
            </button>
          </form>
        ) : (
          <>
            <h1 className={styles.title}>{kb.name}</h1>
            <button
              type="button"
              className={styles.icon}
              aria-label="重新命名"
              onClick={() => setRenaming(true)}
            >
              <Pencil size={16} />
            </button>
          </>
        )}
      </header>

      {error && (
        <div role="alert" className={styles.error}>
          {error}
        </div>
      )}

      <form className={styles.card} onSubmit={upload}>
        <h2 className={styles.cardTitle}>上傳文件</h2>
        <div className={styles.uploadRow}>
          <input
            ref={fileInput}
            type="file"
            accept={ACCEPT}
            multiple
            aria-label="選擇檔案"
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          />
          {needsParser && (
            <select
              aria-label="解析器"
              value={parser}
              onChange={(e) => setParser(e.target.value)}
            >
              {parsers.map((p) => (
                <option key={p.id} value={p.id} disabled={!p.available}>
                  {p.label}
                  {p.available ? '' : '（不可用）'}
                </option>
              ))}
            </select>
          )}
          <button
            type="submit"
            className={styles.primary}
            disabled={
              files.length === 0 || uploading || (needsParser && selectedParser?.available === false)
            }
          >
            <Upload size={16} />
            {uploading ? '上傳中…' : '上傳'}
          </button>
        </div>
        {needsParser &&
          parsers
            .filter((p) => !p.available && p.reason)
            .map((p) => (
              <p key={p.id} className={styles.hint}>
                {p.label.split('（')[0]}：{p.reason}
              </p>
            ))}
      </form>

      <div className={styles.card}>
        <h2 className={styles.cardTitle}>文件</h2>
        {docs !== null && docs.length === 0 ? (
          <p className={styles.muted}>尚未上傳文件</p>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>檔名</th>
                <th>解析器</th>
                <th>大小</th>
                <th>狀態</th>
                <th>區塊</th>
                <th aria-label="操作" />
              </tr>
            </thead>
            <tbody>
              {(docs ?? []).map((d) => (
                <tr key={d.id}>
                  <td className={styles.filename}>{d.filename}</td>
                  <td>{parserLabel(d.parser)}</td>
                  <td>{formatSize(d.size_bytes)}</td>
                  <td>
                    <span className={`${styles.status} ${styles[d.status]}`}>{statusText(d)}</span>
                    {d.error && <div className={styles.docError}>{d.error}</div>}
                  </td>
                  <td>{d.status === 'ready' ? d.chunk_count : '–'}</td>
                  <td className={styles.actions}>
                    {d.status === 'failed' && (
                      <button
                        type="button"
                        className={styles.icon}
                        aria-label={`重試 ${d.filename}`}
                        onClick={() => void retry(d)}
                      >
                        <RotateCcw size={14} />
                      </button>
                    )}
                    <button
                      type="button"
                      className={`${styles.icon} ${styles.danger}`}
                      aria-label={`刪除文件 ${d.filename}`}
                      onClick={() => setPendingDelete(d)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <form className={styles.card} onSubmit={search}>
        <h2 className={styles.cardTitle}>檢索測試</h2>
        <div className={styles.uploadRow}>
          <input
            className={styles.query}
            aria-label="檢索問題"
            placeholder="輸入問題，查看會找到哪些段落"
            value={query}
            maxLength={1000}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" className={styles.primary} disabled={!query.trim() || searching}>
            <Search size={16} />
            檢索
          </button>
        </div>
        {weight !== null && (
          <div className={styles.weightRow}>
            <input
              type="range"
              aria-label="向量權重"
              min={0}
              max={1}
              step={0.05}
              value={weight}
              onChange={(e) => setWeight(Number(e.target.value))}
            />
            <span className={styles.muted}>
              {`向量 ${weight.toFixed(2)} ／ 關鍵字 ${(1 - weight).toFixed(2)}`}
              {weight === defaultWeight && '（聊天預設）'}
            </span>
            {weight !== defaultWeight && (
              <button
                type="button"
                className={styles.secondary}
                onClick={() => setWeight(defaultWeight)}
              >
                還原預設
              </button>
            )}
          </div>
        )}
        {results !== null &&
          (results.length === 0 ? (
            <p className={styles.muted}>沒有找到相關段落</p>
          ) : (
            <SourceCards sources={results} kbId={kb.id} />
          ))}
      </form>

      {pendingDelete && (
        <ConfirmDialog
          message={`確定要刪除「${pendingDelete.filename}」？`}
          onConfirm={confirmDelete}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </div>
  )
}
