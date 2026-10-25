import type { MonitorEventKind } from '../../api/types'

export const KIND_LABEL: Record<MonitorEventKind, string> = {
  document: '文件處理',
  chat: 'AI 對話',
  search: '搜尋測試',
}

export const STATUS_LABEL: Record<string, string> = {
  ok: '完成',
  error: '失敗',
  cancelled: '中斷',
  stopped: '暫停',
  deleted: '已刪除',
}

// in the order the phases happen
export const PHASE_LABEL: Record<string, string> = {
  parse: '解析',
  embed: '向量',
  store: '寫入',
  search: '檢索',
  images: '載入附圖',
  first_token: '第一個回應',
  thinking: '思考',
  answer: '回答',
}

export function phaseEntries(phases: Record<string, number>): [string, number][] {
  const order = Object.keys(PHASE_LABEL)
  return Object.entries(phases).sort(
    ([a], [b]) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99),
  )
}

export function formatSeconds(seconds: number): string {
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`
  if (seconds < 60) return `${seconds.toFixed(1)} 秒`
  const m = Math.floor(seconds / 60)
  return `${m} 分 ${Math.round(seconds - m * 60)} 秒`
}

function pad(n: number): string {
  return String(n).padStart(2, '0')
}

/** Epoch seconds as local HH:MM:SS (or HH:MM). */
export function formatClock(epoch: number, withSeconds = true): string {
  const d = new Date(epoch * 1000)
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`
  return withSeconds ? `${hm}:${pad(d.getSeconds())}` : hm
}

/** Epoch seconds as local MM/DD HH:MM:SS. */
export function formatDateTime(epoch: number): string {
  const d = new Date(epoch * 1000)
  return `${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${formatClock(epoch)}`
}

export function formatMemory(mib: number): string {
  return mib >= 1024 ? `${(mib / 1024).toFixed(2)} GB` : `${Math.round(mib)} MB`
}

export function formatPercent(pct: number | null): string {
  return pct === null ? '—' : `${pct.toFixed(pct < 10 ? 1 : 0)}%`
}

/** "phoebe-embedder-1" -> "embedder"; other containers keep their name. */
export function shortName(source: string, project: string): string {
  if (project !== 'phoebe') return source
  return source.replace(/^phoebe-/, '').replace(/-\d+$/, '')
}

// Categorical slots in their validated order (docs/monitoring.md): each Phoebe service
// keeps its color whatever else is running. Containers of other projects share one
// neutral line ("其他容器"), so there are never more than eight hues.
const SERIES_COLORS = [
  '#2a78d6',
  '#eb6834',
  '#1baf7a',
  '#eda100',
  '#e87ba4',
  '#008300',
  '#6250d6',
  '#e34948',
]
export const SERVICE_ORDER = ['web', 'ingest', 'embedder', 'milvus', 'vllm', 'etcd', 'minio', 'monitor']
export const OTHER_COLOR = '#898781'
export const SINGLE_COLOR = SERIES_COLORS[0]

export function serviceColor(service: string): string {
  const i = SERVICE_ORDER.indexOf(service)
  return i >= 0 ? SERIES_COLORS[i] : OTHER_COLOR
}
