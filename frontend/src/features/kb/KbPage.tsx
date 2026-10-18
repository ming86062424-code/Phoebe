import { useCallback, useEffect, useState } from 'react'
import { BookOpen, Plus, Trash2 } from 'lucide-react'
import { useNavigate, useParams } from 'react-router'
import { api, errorText } from '../../api/client'
import type { KnowledgeBase } from '../../api/types'
import ConfirmDialog from '../../shell/ConfirmDialog'
import KbDetail from './KbDetail'
import styles from './KbPage.module.css'

function parseId(raw: string | undefined): number | null {
  return raw && /^\d+$/.test(raw) ? Number(raw) : null
}

export default function KbPage() {
  const activeId = parseId(useParams()['*'])
  const navigate = useNavigate()
  const [kbs, setKbs] = useState<KnowledgeBase[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<KnowledgeBase | null>(null)

  const reload = useCallback(() => {
    api.listKbs().then(
      (list) => {
        setKbs(list)
        setError(null)
      },
      (e) => setError(errorText(e, '無法載入知識庫')),
    )
  }, [])

  useEffect(reload, [reload])

  const active = kbs?.find((k) => k.id === activeId) ?? null
  useEffect(() => {
    if (kbs === null) return
    // unknown id in the URL, or nothing selected yet: open the first knowledge base
    if (active === null && (activeId !== null || kbs.length > 0)) {
      navigate(kbs.length > 0 ? `/kb/${kbs[0].id}` : '/kb', { replace: true })
    }
  }, [kbs, active, activeId, navigate])

  const create = async (e: React.FormEvent) => {
    e.preventDefault()
    const name = newName.trim()
    if (!name || creating) return
    setCreating(true)
    try {
      const kb = await api.createKb(name)
      setNewName('')
      setKbs((prev) => [kb, ...(prev ?? [])])
      navigate(`/kb/${kb.id}`)
    } catch (e) {
      setError(errorText(e, '建立失敗'))
    } finally {
      setCreating(false)
    }
  }

  const confirmDelete = async () => {
    const target = pendingDelete
    setPendingDelete(null)
    if (!target) return
    try {
      await api.deleteKb(target.id)
      setKbs((prev) => (prev ?? []).filter((k) => k.id !== target.id))
    } catch (e) {
      setError(errorText(e, '刪除失敗'))
    }
  }

  return (
    <div className={styles.page}>
      <nav className={styles.list} aria-label="知識庫列表">
        <form className={styles.create} onSubmit={create}>
          <input
            aria-label="新知識庫名稱"
            placeholder="新知識庫名稱"
            value={newName}
            maxLength={50}
            onChange={(e) => setNewName(e.target.value)}
          />
          <button type="submit" aria-label="新增知識庫" disabled={!newName.trim() || creating}>
            <Plus size={16} />
          </button>
        </form>
        {error && (
          <div role="alert" className={styles.error}>
            {error}
          </div>
        )}
        <ul className={styles.items}>
          {(kbs ?? []).map((k) => (
            <li key={k.id} className={styles.row}>
              <button
                type="button"
                className={`${styles.item} ${k.id === activeId ? styles.active : ''}`}
                aria-current={k.id === activeId ? 'page' : undefined}
                onClick={() => navigate(`/kb/${k.id}`)}
              >
                <span className={styles.name}>{k.name}</span>
                <span className={styles.count}>{k.document_count ?? 0}</span>
              </button>
              <button
                type="button"
                className={styles.delete}
                aria-label={`刪除知識庫 ${k.name}`}
                onClick={() => setPendingDelete(k)}
              >
                <Trash2 size={14} />
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <section className={styles.main}>
        {active ? (
          <KbDetail
            key={active.id}
            kb={active}
            onChanged={(kb) => setKbs((prev) => (prev ?? []).map((k) => (k.id === kb.id ? { ...k, ...kb } : k)))}
            onDocumentsChanged={reload}
          />
        ) : (
          kbs !== null && (
            <div className={styles.empty}>
              <BookOpen size={40} />
              <p>建立第一個知識庫，上傳 PDF 後即可在 AI 對話中引用。</p>
            </div>
          )
        )}
      </section>
      {pendingDelete && (
        <ConfirmDialog
          message={`確定要刪除知識庫「${pendingDelete.name}」與其中所有文件？`}
          onConfirm={confirmDelete}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </div>
  )
}
