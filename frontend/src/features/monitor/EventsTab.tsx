import { CircleAlert, CircleCheck, CircleMinus } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { api, errorText } from '../../api/client'
import type { MonitorEvent, MonitorEventKind } from '../../api/types'
import {
  KIND_LABEL,
  PHASE_LABEL,
  STATUS_LABEL,
  formatDateTime,
  formatSeconds,
  phaseEntries,
} from './format'
import styles from './Monitor.module.css'

const REFRESH_MS = 10000
const PARSER_LABEL: Record<string, string> = { liteparse: 'LiteParse', lightonocr: 'LightOnOCR', text: '文字' }

function StatusBadge({ status }: { status: string }) {
  const Icon = status === 'ok' ? CircleCheck : status === 'error' ? CircleAlert : CircleMinus
  return (
    <span className={styles.statusBadge} data-status={status}>
      <Icon size={14} aria-hidden />
      {STATUS_LABEL[status] ?? status}
    </span>
  )
}

/** The facts that matter for each kind of task, as short "label value" pieces. */
function detailParts(e: MonitorEvent): string[] {
  const d = e.details as Record<string, string | number | boolean | undefined>
  const parts: string[] = []
  if (e.kind === 'document') {
    if (d.parser) parts.push(PARSER_LABEL[String(d.parser)] ?? String(d.parser))
    if (d.pages !== undefined) parts.push(`${d.pages} 頁`)
    if (d.chunks !== undefined) parts.push(`${d.chunks} 個 chunk`)
  } else if (e.kind === 'chat') {
    if (d.model) parts.push(`${d.provider ?? ''} ${d.model}`.trim())
    if (d.kb) parts.push(`知識庫「${d.kb}」`)
    if (d.sources !== undefined) parts.push(`${d.sources} 筆參考資料`)
    if (d.images) parts.push(`${d.images} 張附圖`)
    if (d.regenerate) parts.push('重新產生')
  } else {
    if (d.kb) parts.push(`知識庫「${d.kb}」`)
    if (d.results !== undefined) parts.push(`${d.results} 筆結果`)
    if (typeof d.vector_weight === 'number') parts.push(`向量權重 ${d.vector_weight.toFixed(2)}`)
  }
  if (d.user) parts.push(`使用者 ${d.user}`)
  return parts
}

export default function EventsTab() {
  const [kind, setKind] = useState<MonitorEventKind | ''>('')
  const [events, setEvents] = useState<MonitorEvent[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [more, setMore] = useState<MonitorEvent[]>([])
  const [exhausted, setExhausted] = useState(false)

  const load = useCallback(() => {
    return api.monitorEvents(kind || undefined).then(
      (list) => {
        setEvents(list)
        setError(null)
      },
      (e) => setError(errorText(e, '無法載入事件紀錄')),
    )
  }, [kind])

  useEffect(() => {
    void load()
    const timer = setInterval(() => void load(), REFRESH_MS)
    return () => clearInterval(timer)
  }, [load])

  const shown = events ? [...events, ...more.filter((m) => !events.some((e) => e.id === m.id))] : []

  const loadMore = () => {
    const last = shown[shown.length - 1]
    if (!last) return
    api.monitorEvents(kind || undefined, last.id).then(
      (list) => {
        setMore((m) => [...m, ...list])
        if (list.length === 0) setExhausted(true)
      },
      (e) => setError(errorText(e, '無法載入事件紀錄')),
    )
  }

  return (
    <section className={styles.tabBody} aria-label="事件紀錄">
      <div className={styles.filters}>
        <label className={styles.inline}>
          類型
          <select
            value={kind}
            onChange={(e) => {
              setKind(e.target.value as MonitorEventKind | '')
              setMore([])
              setExhausted(false)
            }}
          >
            <option value="">全部</option>
            {(Object.keys(KIND_LABEL) as MonitorEventKind[]).map((k) => (
              <option key={k} value={k}>
                {KIND_LABEL[k]}
              </option>
            ))}
          </select>
        </label>
        <span className={styles.hint}>每 10 秒自動更新；只記錄時間，不記錄對話內容</span>
      </div>
      {error && <p className={styles.error}>{error}</p>}
      {events && shown.length === 0 && <p className={styles.hint}>還沒有任何紀錄</p>}
      {shown.length > 0 && (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>開始時間</th>
                <th>類型</th>
                <th>狀態</th>
                <th>名稱</th>
                <th className={styles.num}>總時間</th>
                <th>各階段</th>
                <th>詳細</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((e) => (
                <tr key={e.id}>
                  <td className={styles.time}>{formatDateTime(e.started_at)}</td>
                  <td className={styles.nowrap}>{KIND_LABEL[e.kind]}</td>
                  <td>
                    <StatusBadge status={e.status} />
                  </td>
                  <td className={styles.name}>{e.title}</td>
                  <td className={styles.num}>{formatSeconds(e.duration)}</td>
                  <td>
                    <ul className={styles.phases}>
                      {phaseEntries(e.phases).map(([phase, s]) => (
                        <li key={phase}>
                          {PHASE_LABEL[phase] ?? phase} <strong>{formatSeconds(s)}</strong>
                        </li>
                      ))}
                    </ul>
                  </td>
                  <td className={styles.details}>
                    {detailParts(e).join('・')}
                    {typeof e.details.error === 'string' && (
                      <div className={styles.error}>{e.details.error}</div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {events && events.length >= 100 && !exhausted && (
        <button type="button" className={styles.more} onClick={loadMore}>
          載入更早的紀錄
        </button>
      )}
    </section>
  )
}
