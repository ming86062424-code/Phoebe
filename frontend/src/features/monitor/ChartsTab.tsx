import { useCallback, useEffect, useState } from 'react'
import { api, errorText } from '../../api/client'
import type { MonitorSeries } from '../../api/types'
import EventTrack from './EventTrack'
import LineChart, { type ChartPoint, type ChartSeries } from './LineChart'
import {
  OTHER_COLOR,
  SERVICE_ORDER,
  SINGLE_COLOR,
  formatMemory,
  formatPercent,
  serviceColor,
  shortName,
} from './format'
import styles from './Monitor.module.css'

const RANGES = [
  { minutes: 15, label: '15 分鐘' },
  { minutes: 60, label: '1 小時' },
  { minutes: 360, label: '6 小時' },
  { minutes: 1440, label: '24 小時' },
  { minutes: 2880, label: '48 小時' },
]
const REFRESH_MS = 15000
const OTHER = '其他容器'

type Metric = 'mem_mib' | 'cpu_pct' | 'gpu_util_pct'

interface Built {
  memory: ChartSeries[]
  cpu: ChartSeries[]
  total: ChartSeries[]
  vmLimit: number | null
  gpuMemory: ChartSeries[]
  gpuTotal: number | null
  gpuUtil: ChartSeries[]
}

/** Sums one metric of several sources per time bucket (their bucket times differ slightly). */
function sumByBucket(data: MonitorSeries, sources: MonitorSeries['sources'], metric: Metric): ChartPoint[] {
  const buckets = new Map<number, ChartPoint>()
  for (const s of sources)
    for (const p of s.points) {
      const value = p[metric]
      if (value === null) continue
      const b = Math.floor((p.ts - data.since) / data.step)
      const cur = buckets.get(b)
      buckets.set(b, { t: Math.max(cur?.t ?? 0, p.ts), v: (cur?.v ?? 0) + value })
    }
  return [...buckets.entries()].sort(([a], [b]) => a - b).map(([, p]) => p)
}

function points(source: MonitorSeries['sources'][number], metric: Metric): ChartPoint[] {
  return source.points.flatMap((p) => (p[metric] === null ? [] : [{ t: p.ts, v: p[metric] as number }]))
}

function build(data: MonitorSeries): Built {
  const containers = data.sources.filter((s) => s.kind === 'container')
  const ours = containers
    .filter((s) => s.project === 'phoebe')
    .sort(
      (a, b) =>
        (SERVICE_ORDER.indexOf(shortName(a.source, a.project)) + 1 || 99) -
        (SERVICE_ORDER.indexOf(shortName(b.source, b.project)) + 1 || 99),
    )
  const others = containers.filter((s) => s.project !== 'phoebe')
  const perService = (metric: Metric): ChartSeries[] => [
    ...ours.map((s) => {
      const name = shortName(s.source, s.project)
      return { id: s.source, label: name, color: serviceColor(name), points: points(s, metric) }
    }),
    ...(others.length
      ? [{ id: OTHER, label: OTHER, color: OTHER_COLOR, points: sumByBucket(data, others, metric) }]
      : []),
  ]
  const limits = containers.flatMap((s) => s.points.map((p) => p.mem_limit_mib ?? 0))
  const gpus = data.sources.filter((s) => s.kind === 'gpu')
  const gpuTotals = gpus.flatMap((s) => s.points.map((p) => p.mem_limit_mib ?? 0))
  return {
    memory: perService('mem_mib'),
    cpu: perService('cpu_pct'),
    total: containers.length
      ? [{ id: 'total', label: '全部容器', color: SINGLE_COLOR, points: sumByBucket(data, containers, 'mem_mib') }]
      : [],
    vmLimit: limits.length ? Math.max(...limits) : null,
    gpuMemory: gpus.map((s) => ({ id: s.source, label: s.source, color: SINGLE_COLOR, points: points(s, 'mem_mib') })),
    gpuTotal: gpuTotals.length ? Math.max(...gpuTotals) : null,
    gpuUtil: gpus.map((s) => ({ id: s.source, label: s.source, color: SINGLE_COLOR, points: points(s, 'gpu_util_pct') })),
  }
}

export default function ChartsTab() {
  const [minutes, setMinutes] = useState(60)
  // the range the data was loaded for: a newer range dims the charts until it arrives
  const [loaded, setLoaded] = useState<{ minutes: number; data: MonitorSeries } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [hover, setHover] = useState<number | null>(null)

  const load = useCallback(() => {
    return api.monitorSeries(minutes).then(
      (data) => {
        setLoaded({ minutes, data })
        setError(null)
      },
      (e) => setError(errorText(e, '無法載入圖表資料')),
    )
  }, [minutes])

  useEffect(() => {
    void load()
    const timer = setInterval(() => void load(), REFRESH_MS)
    return () => clearInterval(timer)
  }, [load])

  const data = loaded?.data ?? null
  const loading = loaded !== null && loaded.minutes !== minutes
  const built = data ? build(data) : null
  const domain: [number, number] = data ? [data.since, data.until] : [0, 1]
  // a gap longer than a few buckets means the monitor was not sampling: break the line
  const maxGap = data ? Math.max(data.step * 3, 30) : 30
  const common = { domain, hover, onHover: setHover, maxGap }

  return (
    <section className={styles.tabBody} aria-label="圖表">
      <div className={styles.filters} role="group" aria-label="時間範圍">
        {RANGES.map((r) => (
          <button
            key={r.minutes}
            type="button"
            className={r.minutes === minutes ? styles.rangeActive : styles.range}
            aria-pressed={r.minutes === minutes}
            onClick={() => setMinutes(r.minutes)}
          >
            {r.label}
          </button>
        ))}
        <span className={styles.hint}>每 15 秒自動更新；同一時間區間取最大值</span>
      </div>
      {error && <p className={styles.error}>{error}</p>}
      {built && data && (
        <div className={loading ? styles.chartsLoading : styles.charts}>
          <EventTrack events={data.events} domain={domain} hover={hover} />
          <LineChart title="各容器的記憶體" series={built.memory} format={formatMemory}
            unit={1024} {...common} />
          <LineChart
            title="全部容器的記憶體總量（Docker 虛擬機）"
            series={built.total}
            format={formatMemory}
            unit={1024}
            reference={built.vmLimit ? { label: '上限', value: built.vmLimit } : undefined}
            {...common}
          />
          <LineChart title="各容器的 CPU（100% = 一個核心）" series={built.cpu} format={(v) => formatPercent(v)} {...common} />
          {built.gpuMemory.length > 0 ? (
            <>
              <LineChart
                title="GPU 顯存（含 Windows 桌面等其他程式）"
                series={built.gpuMemory}
                format={formatMemory}
            unit={1024}
                reference={built.gpuTotal ? { label: '整張卡', value: built.gpuTotal } : undefined}
                {...common}
              />
              <LineChart
                title="GPU 使用率"
                series={built.gpuUtil}
                format={(v) => formatPercent(v)}
                yMax={100}
                {...common}
              />
            </>
          ) : (
            <p className={styles.hint}>沒有 GPU 資料（沒有 NVIDIA GPU，或 monitor 服務看不到 GPU）</p>
          )}
        </div>
      )}
    </section>
  )
}
