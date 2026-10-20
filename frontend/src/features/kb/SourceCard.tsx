import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, ChevronRight, X } from 'lucide-react'
import type { Source } from '../../api/types'
import styles from './SourceCard.module.css'

const PREVIEW_CHARS = 120
const SCORE_GAP = '　' // full-width space
// same whitelist as the backend image API; anything else stays plain text
const IMAGE_NAME = /^(page|img)_[0-9]+(_[0-9]+)?\.png$/
// ![圖片 i](img:NAME) followed by the description up to the end of the line
const MARKER = /!\[([^\]\n]*)\]\(img:([^)\s]+)\)([^\n]*)/g

export interface RankedSource {
  source: Source
  number: number // the [編號] the model saw: the position in the original list
}

/** Highest hybrid score first; the citation number stays the original position. */
export function rankSources(sources: Source[]): RankedSource[] {
  return sources
    .map((source, i) => ({ source, number: i + 1 }))
    .sort((a, b) => b.source.score - a.source.score || a.number - b.number)
}

function fmt(value: number | null | undefined, digits: number): string {
  return value == null ? '—' : value.toFixed(digits)
}

export function formatScores(s: Source): string {
  return [
    `相關度 ${fmt(s.score, 4)}`,
    `向量 ${fmt(s.vector_score, 3)}`,
    `關鍵字 ${fmt(s.keyword_score, 2)}`,
  ].join(SCORE_GAP)
}

function where(s: Source): string {
  return s.page === 0 ? '全文' : `第 ${s.page} 頁`
}

function imageUrl(kbId: number, documentId: number, name: string): string {
  return `/api/kbs/${kbId}/documents/${documentId}/images/${encodeURIComponent(name)}`
}

function preview(text: string): string {
  const plain = text
    .replace(MARKER, (_, label: string, _name: string, rest: string) => `[${label}]${rest}`)
    .replace(/\s+/g, ' ')
    .trim()
  const chars = Array.from(plain)
  return chars.length > PREVIEW_CHARS ? chars.slice(0, PREVIEW_CHARS).join('') + '…' : plain
}

interface Picture {
  url: string | null
  alt: string
}

function Thumbnail({ picture, inline, onOpen }: { picture: Picture; inline?: boolean; onOpen(p: Picture, el: HTMLElement): void }) {
  const [broken, setBroken] = useState(false)
  if (picture.url === null || broken) {
    return <span className={styles.broken}>{picture.alt}（圖片無法載入）</span>
  }
  return (
    <button
      type="button"
      className={`${styles.thumb} ${inline ? styles.inlineThumb : ''}`}
      aria-label={`放大 ${picture.alt}`}
      onClick={(e) => onOpen(picture, e.currentTarget)}
    >
      <img src={picture.url} alt={picture.alt} loading="lazy" onError={() => setBroken(true)} />
    </button>
  )
}

function Lightbox({ picture, onClose }: { picture: Picture; onClose(): void }) {
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    closeRef.current?.focus()
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      } else if (e.key === 'Tab') {
        // the close button is the only focusable element: keep focus inside
        e.preventDefault()
        closeRef.current?.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = previousOverflow
    }
  }, [onClose])

  return createPortal(
    <div
      className={styles.backdrop}
      data-testid="lightbox-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div role="dialog" aria-modal="true" aria-label={picture.alt} className={styles.dialog}>
        <button ref={closeRef} type="button" className={styles.close} aria-label="關閉" onClick={onClose}>
          <X size={18} />
        </button>
        {picture.url && <img className={styles.full} src={picture.url} alt={picture.alt} />}
      </div>
    </div>,
    document.body,
  )
}

interface Props {
  source: Source
  number: number
  kbId: number | null
}

export default function SourceCard({ source, number, kbId }: Props) {
  const [expanded, setExpanded] = useState(false)
  const [open, setOpen] = useState<Picture | null>(null)
  const triggerRef = useRef<HTMLElement | null>(null)

  // the source's own knowledge base wins: the conversation may have switched since
  const sourceKb = source.kb_id ?? kbId
  const picture = (name: string, alt: string): Picture => ({
    url: sourceKb == null ? null : imageUrl(sourceKb, source.document_id, name),
    alt,
  })
  const show = (p: Picture, el: HTMLElement) => {
    triggerRef.current = el
    setOpen(p)
  }
  const close = useCallback(() => {
    setOpen(null)
    triggerRef.current?.focus()
  }, [])

  const renderBody = (): { body: ReactNode[]; inline: Set<string> } => {
    const body: ReactNode[] = []
    const inline = new Set<string>()
    let last = 0
    for (const m of source.text.matchAll(MARKER)) {
      const [whole, label, name, rest] = m
      if (!IMAGE_NAME.test(name)) continue
      const start = m.index
      if (start > last) body.push(source.text.slice(last, start))
      const description = rest.trim()
      body.push(
        <Thumbnail
          key={`${start}-${name}`}
          inline
          picture={picture(name, description ? `${label} ${description}` : label)}
          onOpen={show}
        />,
      )
      body.push(rest)
      inline.add(name)
      last = start + whole.length
    }
    if (last < source.text.length) body.push(source.text.slice(last))
    return { body, inline }
  }

  let details: ReactNode = null
  if (expanded) {
    const { body, inline } = renderBody()
    const extra = (source.images ?? [])
      .map((img, i) => ({ img, alt: `圖片 ${i + 1}` }))
      .filter(({ img }) => IMAGE_NAME.test(img.name) && !inline.has(img.name))
    const pageImage = source.page_image && IMAGE_NAME.test(source.page_image.name) ? source.page_image : null
    details = (
      <>
        <div className={styles.text} data-testid="source-text">
          {body}
        </div>
        {(pageImage || extra.length > 0) && (
          <div className={styles.gallery}>
            {pageImage && <Thumbnail picture={picture(pageImage.name, where(source))} onOpen={show} />}
            {extra.map(({ img, alt }) => (
              <Thumbnail key={img.name} picture={picture(img.name, alt)} onOpen={show} />
            ))}
          </div>
        )}
      </>
    )
  }

  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <div className={styles.title}>
          [{number}] {source.filename} · {where(source)}
        </div>
        <button
          type="button"
          className={styles.toggle}
          aria-expanded={expanded}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {expanded ? '收合' : '展開'}
        </button>
      </div>
      <div className={styles.scores}>{formatScores(source)}</div>
      {expanded ? (
        details
      ) : (
        <div className={styles.preview} data-testid="source-preview">
          {preview(source.text)}
        </div>
      )}
      {open && <Lightbox picture={open} onClose={close} />}
    </div>
  )
}

export function SourceCards({ sources, kbId }: { sources: Source[]; kbId: number | null }) {
  return (
    <ol className={styles.list}>
      {rankSources(sources).map(({ source, number }) => (
        <li key={`${number}-${source.document_id}-${source.page}`}>
          <SourceCard source={source} number={number} kbId={kbId} />
        </li>
      ))}
    </ol>
  )
}
