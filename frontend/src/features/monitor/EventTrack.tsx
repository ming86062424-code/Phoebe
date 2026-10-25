import { useLayoutEffect, useRef, useState } from 'react'
import type { MonitorEvent, MonitorEventKind } from '../../api/types'
import {
  KIND_LABEL,
  PHASE_LABEL,
  STATUS_LABEL,
  formatClock,
  formatSeconds,
  phaseEntries,
} from './format'
import styles from './Monitor.module.css'

const LANE = 22
const M = { left: 64, right: 16 }
const KINDS: MonitorEventKind[] = ['document', 'chat', 'search']

interface Props {
  events: MonitorEvent[]
  domain: [number, number]
  hover: number | null
}

/** Each task as a bar from its start to its end, one lane per kind, on the charts' time axis. */
export default function EventTrack({ events, domain, hover }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(800)
  const [focused, setFocused] = useState<MonitorEvent | null>(null)
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
  const x = (t: number) => M.left + ((Math.min(Math.max(t, t0), t1) - t0) / Math.max(t1 - t0, 1)) * plotW
  const lanes = KINDS.filter((k) => events.some((e) => e.kind === k))
  const height = Math.max(lanes.length, 1) * LANE + 4

  return (
    <figure className={styles.chart}>
      <figcaption className={styles.chartTitle}>任務</figcaption>
      <div ref={box} className={styles.plot}>
        {lanes.length === 0 ? (
          <p className={styles.hint}>這段時間沒有任務</p>
        ) : (
          <svg width={width} height={height} role="img" aria-label="這段時間的任務">
            {lanes.map((kind, lane) => (
              <g key={kind}>
                <text x={M.left - 8} y={lane * LANE + LANE / 2 + 2} className={styles.tickY}>
                  {KIND_LABEL[kind]}
                </text>
                <line
                  x1={M.left}
                  x2={width - M.right}
                  y1={lane * LANE + LANE / 2 + 2}
                  y2={lane * LANE + LANE / 2 + 2}
                  className={styles.grid}
                />
              </g>
            ))}
            {events.map((e) => {
              const lane = lanes.indexOf(e.kind)
              const start = x(e.started_at)
              const w = Math.max(x(e.ended_at) - start, 4)
              return (
                <rect
                  key={e.id}
                  x={start}
                  y={lane * LANE + 6}
                  width={w}
                  height={LANE - 8}
                  rx={3}
                  tabIndex={0}
                  aria-label={`${KIND_LABEL[e.kind]}：${e.title}，${STATUS_LABEL[e.status] ?? e.status}，${formatSeconds(e.duration)}`}
                  className={e.status === 'ok' ? styles.eventBar : styles.eventBarBad}
                  onPointerEnter={() => setFocused(e)}
                  onPointerLeave={() => setFocused(null)}
                  onFocus={() => setFocused(e)}
                  onBlur={() => setFocused(null)}
                />
              )
            })}
            {hover !== null && hover >= t0 && hover <= t1 && (
              <line x1={x(hover)} x2={x(hover)} y1={0} y2={height} className={styles.crosshair} />
            )}
          </svg>
        )}
        {focused && (
          <div
            className={styles.tooltip}
            style={
              x(focused.started_at) > width / 2
                ? { right: width - x(focused.started_at) + 12 }
                : { left: x(focused.started_at) + 12 }
            }
          >
            <div className={styles.tooltipTime}>
              {formatClock(focused.started_at)}–{formatClock(focused.ended_at)}
            </div>
            <div className={styles.tooltipRow}>
              <strong>{formatSeconds(focused.duration)}</strong>
              <span>
                {KIND_LABEL[focused.kind]}・{STATUS_LABEL[focused.status] ?? focused.status}
              </span>
            </div>
            <div className={styles.tooltipTitle}>{focused.title}</div>
            {phaseEntries(focused.phases).map(([phase, s]) => (
              <div key={phase} className={styles.tooltipRow}>
                <strong>{formatSeconds(s)}</strong>
                <span>{PHASE_LABEL[phase] ?? phase}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  )
}
