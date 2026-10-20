import { useEffect, useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import { api, errorText } from '../../api/client'
import type { DocumentChunk, DocumentChunks, KnowledgeBase } from '../../api/types'
import ChunkBody from './ChunkBody'
import { STATUS_TEXT } from './documentStatus'
import { where } from './SourceCard'
import styles from './ChunkViewer.module.css'

const GAP = '　' // full-width space

// how each parser splits a document and what the vector is made from
const METHODS: Record<string, { name: string; method: string }> = {
  liteparse: {
    name: 'LiteParse',
    method:
      '每頁一塊。文字由 LiteParse 抽取（掃描頁以 Tesseract OCR），PDF 內嵌的圖片另存成圖片；向量由「整頁圖＋該頁文字」產生。',
  },
  lightonocr: {
    name: 'LightOnOCR',
    method:
      '每頁一塊。每頁送給 LightOnOCR（grounding 模式）辨識，去掉頁首、頁尾、頁碼，依座標裁出圖片並附上描述；向量由 OCR 整理出的文字產生。',
  },
  text: {
    name: '文字',
    method: 'txt 依段落合併、md 依標題分段，每塊最多 1500 字（過長的段落再切開）；向量由文字產生。',
  },
}

const KIND_TEXT: Record<DocumentChunk['kind'], string> = {
  page_image: '整頁圖＋文字',
  text: '文字',
}

function stats(chunks: DocumentChunk[]): string {
  const total = chunks.reduce((sum, c) => sum + c.chars, 0)
  const longest = Math.max(...chunks.map((c) => c.chars))
  return [
    `共 ${chunks.length} 塊`,
    `總字數 ${total}`,
    `平均 ${Math.round(total / chunks.length)} 字`,
    `最長 ${longest} 字`,
  ].join(GAP)
}

interface Props {
  kb: KnowledgeBase
  documentId: number
  onBack(): void
}

/** How one document was split: every stored chunk, in order, exactly as retrieval sees it. */
export default function ChunkViewer({ kb, documentId, onBack }: Props) {
  const [data, setData] = useState<DocumentChunks | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let current = true
    api.listChunks(kb.id, documentId).then(
      (d) => current && setData(d),
      (e) => current && setError(errorText(e, '無法讀取切塊')),
    )
    return () => {
      current = false
    }
  }, [kb.id, documentId])

  const doc = data?.document
  const method = doc ? METHODS[doc.parser] : undefined

  return (
    <div className={styles.viewer}>
      <button type="button" className={styles.back} onClick={onBack} aria-label={`返回 ${kb.name}`}>
        <ArrowLeft size={16} />
        {kb.name}
      </button>
      {error && (
        <div role="alert" className={styles.error}>
          {error}
        </div>
      )}
      {doc && data && (
        <>
          <header className={styles.summary}>
            <h2 className={styles.title}>{doc.filename}</h2>
            <p className={styles.meta}>
              解析器：{method?.name ?? doc.parser}
              {GAP}狀態：{STATUS_TEXT[doc.status]}
            </p>
            {method && <p className={styles.method}>{method.method}</p>}
            {data.chunks.length > 0 && <p className={styles.stats}>{stats(data.chunks)}</p>}
          </header>

          {data.chunks.length === 0 ? (
            <p className={styles.empty}>
              {doc.status === 'failed'
                ? `這份文件處理失敗：${doc.error ?? '原因不明'}`
                : doc.status === 'ready'
                  ? '這份文件沒有任何切塊。'
                  : `這份文件還沒有處理完成（${STATUS_TEXT[doc.status]}），完成後才會有切塊。`}
            </p>
          ) : (
            <ol className={styles.list}>
              {data.chunks.map((c) => (
                <li key={c.index}>
                  <article className={styles.chunk} aria-label={`第 ${c.index} 塊`}>
                    <div className={styles.chunkHead}>
                      {[`#${c.index}`, where(c.page), KIND_TEXT[c.kind], `${c.chars} 字`].join(' · ')}
                    </div>
                    <ChunkBody
                      text={c.text}
                      documentId={doc.id}
                      kbId={kb.id}
                      images={c.images}
                      pageImage={c.page_image}
                      pageLabel={where(c.page)}
                    />
                  </article>
                </li>
              ))}
            </ol>
          )}
        </>
      )}
    </div>
  )
}
