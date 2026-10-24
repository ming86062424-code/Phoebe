import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type { Source } from '../../api/types'
import ChunkBody, { MARKER } from './ChunkBody'
import styles from './SourceCard.module.css'

const PREVIEW_CHARS = 120
const SCORE_GAP = '　' // full-width space

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

export function where(page: number): string {
  return page === 0 ? '全文' : `第 ${page} 頁`
}

function preview(text: string): string {
  const plain = text
    .replace(MARKER, (_, label: string, _name: string, rest: string) => `[${label}]${rest}`)
    .replace(/\s+/g, ' ')
    .trim()
  const chars = Array.from(plain)
  return chars.length > PREVIEW_CHARS ? chars.slice(0, PREVIEW_CHARS).join('') + '…' : plain
}

interface Props {
  source: Source
  number: number
  kbId: number | null
}

export default function SourceCard({ source, number, kbId }: Props) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <div className={styles.title}>
          [{number}] {source.filename} · {where(source.page)}
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
        <ChunkBody
          text={source.text}
          documentId={source.document_id}
          // the source's own knowledge base wins: the conversation may have switched since
          kbId={source.kb_id ?? kbId}
          images={source.images}
          pageImage={source.page_image}
          pageLabel={where(source.page)}
          testId="source-text"
        />
      ) : (
        <div className={styles.preview} data-testid="source-preview">
          {preview(source.text)}
        </div>
      )}
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
