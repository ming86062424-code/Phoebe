import { useLayoutEffect, useRef, useState, type PointerEvent } from 'react'
import { formatClock } from './format'
import styles from './Monitor.module.css'

export interface ChartPoint {
  t: number // epoch seconds
  v: number
}

export interface ChartSeries {
  id: string
  label: string
  color: string
  points: ChartPoint[]
}

interface Props {
  title: string
  series: ChartSeries[]
  domain: [number, number]
  format: (v: number) => string
  /** a fixed top of the scale (e.g. 100 for percentages); otherwise from the data */
  yMax?: number
  /** a horizontal reference, e.g. the memory limit */
  reference?: { label: string; value: number }
  /** the time under the pointer, shared by every chart on the page */
  hover: number | null
  onHover: (t: number | null) => void
  /** the widest gap between two points still drawn as one line */
  maxGap: number
  /** the axis rounds its top in this unit (1024 for MiB values shown in GB) */
  unit?: number
}

const HEIGHT = 200
const M = { top: 12, right: 16, bottom: 24, left: 64 }

function niceMax(v: number): number {
  if (v <= 0) return 1
  const exp = 10 ** Math.floor(Math.log10(v))
  const f = v / exp
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10
  return nice * exp
}

function nearest(points: ChartPoint[], t: number, within: number): ChartPoint | null {
  let best: ChartPoint | null = null
  for (const p of points) {
    if (Math.abs(p.t - t) <= within && (!best || Math.abs(p.t - t) < Math.abs(best.t - t))) best = p
  }
  return best
}

export default function LineChart({
  title,
  series,
  domain,
  format,
  yMax,
  reference,
  hover,
  onHover,
  maxGap,
  unit = 1,
}: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(800)
  useLayoutEffect(() => {
    const el = box.current
    if (!el) return
    const measure = () => setWidth(Math.max(320, el.clientWidth))
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const [t0, t1] = domain
  const plotW = width - M.left - M.right
  const plotH = HEIGHT - M.top - M.bottom
  const dataMax = Math.max(
    0,
    reference?.value ?? 0,
    ...series.flatMap((s) => s.points.map((p) => p.v)),
  )
  const top = yMax ?? niceMax((dataMax * 1.05) / unit) * unit
  const x = (t: number) => M.left + ((t - t0) / Math.max(t1 - t0, 1)) * plotW
  const y = (v: number) => M.top + plotH - (Math.min(v, top) / top) * plotH
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * top)
  const xTicks = Array.from({ length: 5 }, (_, i) => t0 + ((t1 - t0) * i) / 4)

  function path(points: ChartPoint[]): string {
    let d = ''
    let prev: ChartPoint | null = null
    for (const p of points) {
      const cmd = prev && p.t - prev.t <= maxGap ? 'L' : 'M'
      d += `${cmd}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`
      prev = p
    }
    return d
  }

  function move(e: PointerEvent<SVGRectElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    const t = t0 + ((e.clientX - rect.left) / rect.width) * (t1 - t0)
    // snap to the closest sample time of any series
    let best: number | null = null
    for (const s of series)
      for (const p of s.points) if (best === null || Math.abs(p.t - t) < Math.abs(best - t)) best = p.t
    onHover(best)
  }

  const readout =
    hover === null
      ? []
      : series
          .map((s) => ({ s, p: nearest(s.points, hover, maxGap) }))
          .filter((r): r is { s: ChartSeries; p: ChartPoint } => r.p !== null)
          .sort((a, b) => b.p.v - a.p.v)
  const hoverX = hover === null ? null : x(hover)
  const empty = series.every((s) => s.points.length === 0)

  return (
    <figure className={styles.chart}>
      <figcaption className={styles.chartTitle}>{title}</figcaption>
      {series.length > 1 && (
        <ul className={styles.legend} aria-label={`${title}的圖例`}>
          {series.map((s) => (
            <li key={s.id}>
              <span className={styles.lineKey} style={{ background: s.color }} />
              {s.label}
            </li>
          ))}
        </ul>
      )}
      <div ref={box} className={styles.plot}>
        {empty ? (
          <p className={styles.hint}>這段時間沒有資料</p>
        ) : (
          <svg width={width} height={HEIGHT} role="img" aria-label={title}>
            {yTicks.map((v) => (
              <g key={v}>
                <line
                  x1={M.left}
                  x2={width - M.right}
                  y1={y(v)}
                  y2={y(v)}
                  className={v === 0 ? styles.axis : styles.grid}
                />
                <text x={M.left - 8} y={y(v)} className={styles.tickY}>
                  {format(v)}
                </text>
              </g>
            ))}
            {xTicks.map((t, i) => (
              <text
                key={t}
                x={x(t)}
                y={HEIGHT - 6}
                className={styles.tickX}
                // the outer labels stay inside the chart
                textAnchor={i === 0 ? 'start' : i === xTicks.length - 1 ? 'end' : 'middle'}
              >
                {formatClock(t, t1 - t0 < 1800)}
              </text>
            ))}
            {reference && (
              <g>
                <line
                  x1={M.left}
                  x2={width - M.right}
                  y1={y(reference.value)}
                  y2={y(reference.value)}
                  className={styles.reference}
                />
                <text x={M.left + 4} y={y(reference.value) - 4} className={styles.referenceLabel}>
                  {reference.label} {format(reference.value)}
                </text>
              </g>
            )}
            {series.map((s) => (
              <path key={s.id} d={path(s.points)} stroke={s.color} className={styles.line} />
            ))}
            {hoverX !== null && (
              <g>
                <line x1={hoverX} x2={hoverX} y1={M.top} y2={M.top + plotH} className={styles.crosshair} />
                {readout.map(({ s, p }) => (
                  <circle key={s.id} cx={x(p.t)} cy={y(p.v)} r={4} fill={s.color} className={styles.dot} />
                ))}
              </g>
            )}
            <rect
              x={M.left}
              y={M.top}
              width={plotW}
              height={plotH}
              fill="transparent"
              onPointerMove={move}
              onPointerLeave={() => onHover(null)}
            />
          </svg>
        )}
        {hover !== null && readout.length > 0 && hoverX !== null && (
          <div
            className={styles.tooltip}
            style={hoverX > width / 2 ? { right: width - hoverX + 12 } : { left: hoverX + 12 }}
          >
            <div className={styles.tooltipTime}>{formatClock(hover)}</div>
            {readout.map(({ s, p }) => (
              <div key={s.id} className={styles.tooltipRow}>
                <span className={styles.lineKey} style={{ background: s.color }} />
                <strong>{format(p.v)}</strong>
                <span>{s.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  )
}
