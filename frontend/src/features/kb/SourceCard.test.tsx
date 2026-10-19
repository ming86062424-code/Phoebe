import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import type { Source } from '../../api/types'
import SourceCard, { SourceCards, formatScores, rankSources } from './SourceCard'

function source(overrides: Partial<Source> = {}): Source {
  return {
    document_id: 17,
    filename: '手冊.pdf',
    page: 2,
    score: 0.0328,
    vector_score: 0.81234,
    keyword_score: 2.5512,
    text: '員工每年享有特休假十四天',
    kind: 'page_image',
    images: [],
    page_image: null,
    ...overrides,
  }
}

const imgSrc = (name: string) => `/api/kbs/3/documents/17/images/${name}`

describe('rankSources', () => {
  it('sorts by score descending and keeps the original citation numbers', () => {
    const ranked = rankSources([
      source({ filename: 'a', score: 0.01 }),
      source({ filename: 'b', score: 0.03 }),
      source({ filename: 'c', score: 0.02 }),
    ])
    expect(ranked.map((r) => [r.number, r.source.filename])).toEqual([
      [2, 'b'],
      [3, 'c'],
      [1, 'a'],
    ])
  })

  it('keeps the original order for equal scores', () => {
    const ranked = rankSources([source({ filename: 'a' }), source({ filename: 'b' })])
    expect(ranked.map((r) => r.number)).toEqual([1, 2])
  })
})

describe('formatScores', () => {
  it('formats relevance, vector and keyword scores', () => {
    expect(formatScores(source())).toBe('相關度 0.0328　向量 0.812　關鍵字 2.55')
  })

  it('shows a dash for missing scores', () => {
    expect(formatScores(source({ vector_score: null, keyword_score: undefined }))).toBe(
      '相關度 0.0328　向量 —　關鍵字 —',
    )
  })
})

describe('SourceCards', () => {
  it('renders the cards in score order', () => {
    render(
      <SourceCards
        kbId={3}
        sources={[
          source({ filename: 'low.pdf', score: 0.01 }),
          source({ filename: 'high.pdf', score: 0.05 }),
        ]}
      />,
    )
    const items = screen.getAllByRole('listitem')
    expect(items[0]).toHaveTextContent('[2] high.pdf · 第 2 頁')
    expect(items[1]).toHaveTextContent('[1] low.pdf · 第 2 頁')
  })
})

describe('SourceCard', () => {
  it('cards are collapsed by default and show a 120-char preview', () => {
    const long = '甲'.repeat(100) + '乙'.repeat(100)
    render(<SourceCard source={source({ text: long })} number={1} kbId={3} />)
    expect(screen.getByText('[1] 手冊.pdf · 第 2 頁')).toBeInTheDocument()
    expect(screen.getByText(/^相關度/).textContent).toBe('相關度 0.0328　向量 0.812　關鍵字 2.55')
    const preview = screen.getByTestId('source-preview')
    expect(preview.textContent).toBe('甲'.repeat(100) + '乙'.repeat(20) + '…')
    expect(screen.queryByText(long)).toBeNull()
    expect(screen.getByRole('button', { name: '展開' })).toHaveAttribute('aria-expanded', 'false')
  })

  it('shows a short text in full without an ellipsis', () => {
    render(<SourceCard source={source()} number={1} kbId={3} />)
    expect(screen.getByTestId('source-preview').textContent).toBe('員工每年享有特休假十四天')
  })

  it('shows 全文 for text files', () => {
    render(<SourceCard source={source({ filename: 'a.md', page: 0, kind: 'text' })} number={4} kbId={3} />)
    expect(screen.getByText('[4] a.md · 全文')).toBeInTheDocument()
  })

  it('expands to the full text and thumbnails, then collapses', async () => {
    const long = '甲'.repeat(200)
    render(
      <SourceCard
        source={source({
          text: long,
          page_image: { name: 'page_2.png', width: 1000, height: 1400 },
          images: [{ name: 'img_2_1.png', width: 100, height: 100 }],
        })}
        number={1}
        kbId={3}
      />,
    )
    expect(screen.queryByRole('img')).toBeNull()
    await userEvent.click(screen.getByRole('button', { name: '展開' }))
    expect(screen.getByText(long)).toBeInTheDocument()
    expect(screen.queryByTestId('source-preview')).toBeNull()
    const page = screen.getByRole('img', { name: '第 2 頁' })
    expect(page).toHaveAttribute('src', imgSrc('page_2.png'))
    expect(page).toHaveAttribute('loading', 'lazy')
    expect(screen.getByRole('img', { name: '圖片 1' })).toHaveAttribute('src', imgSrc('img_2_1.png'))
    await userEvent.click(screen.getByRole('button', { name: '收合' }))
    expect(screen.queryByText(long)).toBeNull()
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('renders img: markers as inline images and does not repeat them below', async () => {
    const text = '說明如下\n![圖片 1](img:img_2_1.png) 組織架構圖\n其他內容'
    render(
      <SourceCard
        source={source({
          text,
          images: [
            { name: 'img_2_1.png', width: 100, height: 100 },
            { name: 'img_2_2.png', width: 100, height: 100 },
          ],
        })}
        number={1}
        kbId={3}
      />,
    )
    // the preview shows the label instead of the raw marker
    expect(screen.getByTestId('source-preview').textContent).not.toContain('img:')
    await userEvent.click(screen.getByRole('button', { name: '展開' }))
    const inline = screen.getByRole('img', { name: '圖片 1 組織架構圖' })
    expect(inline).toHaveAttribute('src', imgSrc('img_2_1.png'))
    const body = screen.getByTestId('source-text')
    expect(body).toContainElement(inline)
    expect(body.textContent).toContain('說明如下')
    expect(body.textContent).toContain('組織架構圖')
    expect(body.textContent).not.toContain('img:')
    const all = screen.getAllByRole('img')
    expect(all.map((i) => i.getAttribute('src'))).toEqual([imgSrc('img_2_1.png'), imgSrc('img_2_2.png')])
  })

  it('leaves markers with an unexpected name as plain text', async () => {
    const text = '![圖片 1](img:../app.db) 壞的'
    render(<SourceCard source={source({ text })} number={1} kbId={3} />)
    await userEvent.click(screen.getByRole('button', { name: '展開' }))
    expect(screen.queryByRole('img')).toBeNull()
    expect(screen.getByTestId('source-text').textContent).toBe(text)
  })

  it('falls back gracefully when an image fails to load', async () => {
    render(
      <SourceCard
        source={source({ page_image: { name: 'page_2.png', width: 10, height: 10 } })}
        number={1}
        kbId={3}
      />,
    )
    await userEvent.click(screen.getByRole('button', { name: '展開' }))
    fireEvent.error(screen.getByRole('img', { name: '第 2 頁' }))
    expect(screen.queryByRole('img', { name: '第 2 頁' })).toBeNull()
    expect(screen.getByText('第 2 頁（圖片無法載入）')).toBeInTheDocument()
  })

  it('does not request images without a knowledge base id', async () => {
    render(
      <SourceCard
        source={source({ page_image: { name: 'page_2.png', width: 10, height: 10 } })}
        number={1}
        kbId={null}
      />,
    )
    await userEvent.click(screen.getByRole('button', { name: '展開' }))
    expect(screen.queryByRole('img')).toBeNull()
    expect(screen.getByText('第 2 頁（圖片無法載入）')).toBeInTheDocument()
  })

  it('tolerates sources saved before images existed', async () => {
    const legacy = { document_id: 1, filename: 'old.pdf', page: 1, score: 0.5, text: '舊資料' } as Source
    render(<SourceCard source={legacy} number={1} kbId={3} />)
    expect(screen.getByText(/^相關度/).textContent).toBe('相關度 0.5000　向量 —　關鍵字 —')
    await userEvent.click(screen.getByRole('button', { name: '展開' }))
    expect(screen.getByTestId('source-text').textContent).toBe('舊資料')
  })
})

describe('lightbox', () => {
  async function openLightbox() {
    render(
      <SourceCard
        source={source({ page_image: { name: 'page_2.png', width: 10, height: 10 } })}
        number={1}
        kbId={3}
      />,
    )
    await userEvent.click(screen.getByRole('button', { name: '展開' }))
    const trigger = screen.getByRole('button', { name: '放大 第 2 頁' })
    await userEvent.click(trigger)
    return trigger
  }

  it('opens an enlarged image in a modal dialog with focus on the close button', async () => {
    await openLightbox()
    const dialog = screen.getByRole('dialog', { name: '第 2 頁' })
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(within(dialog).getByRole('img', { name: '第 2 頁' })).toHaveAttribute('src', imgSrc('page_2.png'))
    expect(within(dialog).getByRole('button', { name: '關閉' })).toHaveFocus()
  })

  it('closes on Escape and returns focus to the thumbnail', async () => {
    const trigger = await openLightbox()
    await userEvent.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(trigger).toHaveFocus()
  })

  it('closes on a backdrop click but not on an image click', async () => {
    await openLightbox()
    const dialog = screen.getByRole('dialog')
    await userEvent.click(within(dialog).getByRole('img'))
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    await userEvent.click(screen.getByTestId('lightbox-backdrop'))
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('closes with the close button', async () => {
    await openLightbox()
    await userEvent.click(screen.getByRole('button', { name: '關閉' }))
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
