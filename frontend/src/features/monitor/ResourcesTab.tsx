import { useCallback, useEffect, useState } from 'react'
import { api, errorText } from '../../api/client'
import type { MonitorStatus, ResourceSample } from '../../api/types'
import { formatClock, formatDateTime, formatMemory, formatPercent, shortName } from './format'
import styles from './Monitor.module.css'

const REFRESH_MS = 10000
const SAMPLE_RANGES = [
  { minutes: 5, label: '5 分鐘' },
  { minutes: 15, label: '15 分鐘' },
  { minutes: 60, label: '1 小時' },
]

function share(s: ResourceSample): string {
  return s.mem_limit_mib ? `${((s.mem_mib / s.mem_limit_mib) * 100).toFixed(0)}%` : '—'
}

function CollectorState({ status }: { status: MonitorStatus }) {
  const c = status.collector
  if (status.running && c) {
    return (
      <p className={styles.hint}>
        monitor 服務運作中，每 {c.interval_seconds} 秒取樣一次，最後取樣 {formatClock(c.last_sample_at ?? 0)}
        {c.error && <span className={styles.error}>（{c.error}）</span>}
      </p>
    )
  }
  return (
    <div className={styles.warning} role="status">
      <strong>monitor 服務沒有在取樣</strong>
      <span>
        {c?.last_sample_at
          ? `最後一次取樣在 ${formatDateTime(c.last_sample_at)}。`
          : '還沒有任何取樣紀錄。'}
        資源用量由 Docker 的 monitor 服務記錄（cd docker && docker compose up -d monitor）；本機開發模式只會記錄任務。
        {c?.error && ` 最後的錯誤：${c.error}`}
      </span>
    </div>
  )
}

export default function ResourcesTab() {
  const [status, setStatus] = useState<MonitorStatus | null>(null)
  const [minutes, setMinutes] = useState(15)
  const [source, setSource] = useState('')
  const [samples, setSamples] = useState<ResourceSample[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    return Promise.all([api.monitorStatus(), api.monitorSamples(minutes, source || undefined)]).then(
      ([s, list]) => {
        setStatus(s)
        setSamples(list)
        setError(null)
      },
      (e) => setError(errorText(e, '無法載入資源用量')),
    )
  }, [minutes, source])

  useEffect(() => {
    void load()
    const timer = setInterval(() => void load(), REFRESH_MS)
    return () => clearInterval(timer)
  }, [load])

  const containers = status?.latest.filter((s) => s.kind === 'container') ?? []
  const gpus = status?.latest.filter((s) => s.kind === 'gpu') ?? []
  const total = containers.reduce((sum, s) => sum + s.mem_mib, 0)
  const vmLimit = Math.max(0, ...containers.map((s) => s.mem_limit_mib ?? 0))
  const sources = status?.latest.map((s) => s.source) ?? []

  return (
    <section className={styles.tabBody} aria-label="資源使用">
      {error && <p className={styles.error}>{error}</p>}
      {status && <CollectorState status={status} />}

      {status && status.latest.length > 0 && (
        <>
          <h2 className={styles.sectionTitle}>
            目前用量（{formatClock(status.latest[0].ts)}）
          </h2>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>來源</th>
                  <th>專案</th>
                  <th className={styles.num}>記憶體</th>
                  <th className={styles.num}>佔上限</th>
                  <th className={styles.num}>CPU</th>
                  <th className={styles.num}>GPU 使用率</th>
                </tr>
              </thead>
              <tbody>
                {containers.map((s) => (
                  <tr key={s.source}>
                    <td className={styles.name}>{shortName(s.source, s.project)}</td>
                    <td>{s.project || '—'}</td>
                    <td className={styles.num}>{formatMemory(s.mem_mib)}</td>
                    <td className={styles.num}>{share(s)}</td>
                    <td className={styles.num}>{formatPercent(s.cpu_pct)}</td>
                    <td className={styles.num}>—</td>
                  </tr>
                ))}
                {containers.length > 0 && (
                  <tr className={styles.totalRow}>
                    <td>全部容器</td>
                    <td>—</td>
                    <td className={styles.num}>{formatMemory(total)}</td>
                    <td className={styles.num}>
                      {vmLimit ? `${((total / vmLimit) * 100).toFixed(0)}%（上限 ${formatMemory(vmLimit)}）` : '—'}
                    </td>
                    <td className={styles.num}>
                      {formatPercent(containers.reduce((sum, s) => sum + (s.cpu_pct ?? 0), 0))}
                    </td>
                    <td className={styles.num}>—</td>
                  </tr>
                )}
                {gpus.map((s) => (
                  <tr key={s.source}>
                    <td>{s.source} 顯存</td>
                    <td>—</td>
                    <td className={styles.num}>{formatMemory(s.mem_mib)}</td>
                    <td className={styles.num}>
                      {share(s)}（共 {formatMemory(s.mem_limit_mib ?? 0)}）
                    </td>
                    <td className={styles.num}>—</td>
                    <td className={styles.num}>{formatPercent(s.gpu_util_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className={styles.hint}>
            容器的記憶體是 Docker 虛擬機的主機記憶體（算法同 docker stats），所有容器共用同一個上限，包括其他專案的容器。
            CPU 的 100% 等於一個核心。GPU 顯存包含 Windows 桌面等 Docker 以外的程式。
          </p>
        </>
      )}

      <h2 className={styles.sectionTitle}>詳細取樣紀錄</h2>
      <div className={styles.filters}>
        <label className={styles.inline}>
          時間
          <select value={minutes} onChange={(e) => setMinutes(Number(e.target.value))}>
            {SAMPLE_RANGES.map((r) => (
              <option key={r.minutes} value={r.minutes}>
                最近 {r.label}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.inline}>
          來源
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">全部</option>
            {sources.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <span className={styles.hint}>新的在前，最多 500 筆</span>
      </div>
      {samples && samples.length === 0 && <p className={styles.hint}>這段時間沒有取樣</p>}
      {samples && samples.length > 0 && (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>時間</th>
                <th>來源</th>
                <th className={styles.num}>記憶體</th>
                <th className={styles.num}>上限</th>
                <th className={styles.num}>CPU</th>
                <th className={styles.num}>GPU 使用率</th>
              </tr>
            </thead>
            <tbody>
              {samples.map((s) => (
                <tr key={`${s.ts}-${s.source}`}>
                  <td className={styles.time}>{formatClock(s.ts)}</td>
                  <td className={styles.name}>{s.source}</td>
                  <td className={styles.num}>{formatMemory(s.mem_mib)}</td>
                  <td className={styles.num}>{s.mem_limit_mib ? formatMemory(s.mem_limit_mib) : '—'}</td>
                  <td className={styles.num}>{formatPercent(s.cpu_pct)}</td>
                  <td className={styles.num}>{formatPercent(s.gpu_util_pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
