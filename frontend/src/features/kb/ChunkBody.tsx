import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import type { SourceImage } from '../../api/types'
import styles from './ChunkBody.module.css'

// same whitelist as the backend image API; anything else stays plain text
export const IMAGE_NAME = /^(page|img)_[0-9]+(_[0-9]+)?\.png$/
// ![圖片 i](img:NAME) followed by the description up to the end of the line
export const MARKER = /!\[([^\]\n]*)\]\(img:([^)\s]+)\)([^\n]*)/g

export function imageUrl(kbId: number, documentId: number, name: string): string {
  return `/api/kbs/${kbId}/documents/${documentId}/images/${encodeURIComponent(name)}`
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
  text: string
  documentId: number
  kbId: number | null // null: the images can't be fetched (shown as unavailable)
  images?: SourceImage[]
  pageImage?: SourceImage | null
  pageLabel: string // alt text of the page image, e.g. 「第 2 頁」
  testId?: string
}

/**
 * A chunk's full text with its `![圖片 i](img:NAME)` markers shown as the pictures,
 * then the page image and any picture the text does not mention; a click enlarges one.
 */
export default function ChunkBody({ text, documentId, kbId, images, pageImage, pageLabel, testId }: Props) {
  const [open, setOpen] = useState<Picture | null>(null)
  const triggerRef = useRef<HTMLElement | null>(null)

  const picture = (name: string, alt: string): Picture => ({
    url: kbId == null ? null : imageUrl(kbId, documentId, name),
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

  const body: ReactNode[] = []
  const inline = new Set<string>()
  let last = 0
  for (const m of text.matchAll(MARKER)) {
    const [whole, label, name, rest] = m
    if (!IMAGE_NAME.test(name)) continue
    const start = m.index
    if (start > last) body.push(text.slice(last, start))
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
  if (last < text.length) body.push(text.slice(last))

  const extra = (images ?? [])
    .map((img, i) => ({ img, alt: `圖片 ${i + 1}` }))
    .filter(({ img }) => IMAGE_NAME.test(img.name) && !inline.has(img.name))
  const page = pageImage && IMAGE_NAME.test(pageImage.name) ? pageImage : null

  return (
    <>
      <div className={styles.text} data-testid={testId}>
        {body}
      </div>
      {(page || extra.length > 0) && (
        <div className={styles.gallery}>
          {page && <Thumbnail picture={picture(page.name, pageLabel)} onOpen={show} />}
          {extra.map(({ img, alt }) => (
            <Thumbnail key={img.name} picture={picture(img.name, alt)} onOpen={show} />
          ))}
        </div>
      )}
      {open && <Lightbox picture={open} onClose={close} />}
    </>
  )
}
