import { createEvent, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ApiError } from '../../api/client'
import type { ParserInfo } from '../../api/types'
import UploadPanel from './UploadPanel'

const parsers: ParserInfo[] = [
  { id: 'liteparse', label: 'LiteParse（快速）', available: true, reason: '' },
  { id: 'lightonocr', label: 'LightOnOCR-3（GPU OCR）', available: false, reason: 'OCR 服務未啟動' },
]

const pdf = (name = 'a.pdf') => new File(['%PDF-1.4'], name, { type: 'application/pdf' })

function setup(upload = vi.fn().mockResolvedValue(undefined), parser: ParserInfo | null = parsers[0]) {
  const onUploaded = vi.fn()
  render(<UploadPanel parser={parser} upload={upload} onUploaded={onUploaded} />)
  return { upload, onUploaded }
}

function drop(files: File[]) {
  const zone = screen.getByRole('button', { name: /拖曳檔案到這裡/ })
  fireEvent.dragOver(zone, { dataTransfer: { files, types: ['Files'] } })
  fireEvent.drop(zone, { dataTransfer: { files, types: ['Files'] } })
}

function row(name: string) {
  return screen.getByRole('listitem', { name })
}

describe('UploadPanel', () => {
  it('uploads every chosen file right away, one after another', async () => {
    const { upload, onUploaded } = setup()
    await userEvent.upload(screen.getByLabelText('選擇檔案'), [pdf('a.pdf'), pdf('b.pdf')])
    await waitFor(() => expect(within(row('b.pdf')).getByText('已上傳')).toBeInTheDocument())
    expect(within(row('a.pdf')).getByText('已上傳')).toBeInTheDocument()
    expect(upload.mock.calls.map(([f]) => f.name)).toEqual(['a.pdf', 'b.pdf'])
    expect(onUploaded).toHaveBeenCalledTimes(2)
  })

  it('starts uploading files dropped on the zone', async () => {
    const { upload } = setup()
    drop([pdf('drop.pdf'), new File(['# 標題'], 'notes.md')])
    await waitFor(() => expect(within(row('notes.md')).getByText('已上傳')).toBeInTheDocument())
    expect(upload.mock.calls.map(([f]) => f.name)).toEqual(['drop.pdf', 'notes.md'])
  })

  it('highlights the zone while files are dragged over it', () => {
    setup()
    const zone = screen.getByRole('button', { name: /拖曳檔案到這裡/ })
    fireEvent.dragEnter(zone, { dataTransfer: { types: ['Files'] } })
    expect(zone).toHaveAttribute('data-dragging', 'true')
    fireEvent.dragLeave(zone, { dataTransfer: { types: ['Files'] } })
    expect(zone).toHaveAttribute('data-dragging', 'false')
  })

  it("says which parser the knowledge base uses, without offering to change it here", () => {
    setup(undefined, { ...parsers[1], available: true })
    expect(screen.getByText('PDF 與圖片會用「LightOnOCR-3」解析（在上方的知識庫設定中更改）')).toBeInTheDocument()
    expect(screen.queryByRole('combobox')).toBeNull()
  })

  it('rejects unsupported files without sending them', async () => {
    const { upload } = setup()
    drop([new File(['MZ'], 'tool.exe'), pdf('ok.pdf')])
    await waitFor(() => expect(within(row('ok.pdf')).getByText('已上傳')).toBeInTheDocument())
    expect(within(row('tool.exe')).getByText(/不支援的檔案類型/)).toBeInTheDocument()
    expect(upload.mock.calls.map(([f]) => f.name)).toEqual(['ok.pdf'])
  })

  it('shows the server error of a failed file and goes on with the rest', async () => {
    const upload = vi
      .fn()
      .mockRejectedValueOnce(new ApiError(422, '無法讀取檔案內容'))
      .mockResolvedValue(undefined)
    setup(upload)
    drop([pdf('bad.pdf'), pdf('good.pdf')])
    await waitFor(() => expect(within(row('good.pdf')).getByText('已上傳')).toBeInTheDocument())
    expect(within(row('bad.pdf')).getByText('無法讀取檔案內容')).toBeInTheDocument()
  })

  it("does not send PDFs or images while the knowledge base's parser is unavailable, but text files go", async () => {
    const { upload } = setup(undefined, parsers[1])
    expect(screen.getByText(/LightOnOCR-3：OCR 服務未啟動/)).toBeInTheDocument()
    drop([pdf('scan.pdf'), new File(['x'], 'b.TXT')])
    await waitFor(() => expect(within(row('b.TXT')).getByText('已上傳')).toBeInTheDocument())
    expect(within(row('scan.pdf')).getByText(/目前無法使用/)).toBeInTheDocument()
    expect(upload.mock.calls.map(([f]) => f.name)).toEqual(['b.TXT'])
  })

  it('queues files dropped while an upload is running', async () => {
    let release!: () => void
    const upload = vi
      .fn()
      .mockImplementationOnce(() => new Promise<void>((r) => (release = r)))
      .mockResolvedValue(undefined)
    setup(upload)
    drop([pdf('first.pdf')])
    await waitFor(() => expect(within(row('first.pdf')).getByText('上傳中…')).toBeInTheDocument())
    drop([pdf('second.pdf')])
    expect(within(row('second.pdf')).getByText('等待中')).toBeInTheDocument()
    expect(upload).toHaveBeenCalledTimes(1)
    release()
    await waitFor(() => expect(within(row('second.pdf')).getByText('已上傳')).toBeInTheDocument())
  })

  it('clears finished rows', async () => {
    setup()
    drop([pdf()])
    await waitFor(() => expect(within(row('a.pdf')).getByText('已上傳')).toBeInTheDocument())
    await userEvent.click(screen.getByRole('button', { name: '清除已完成' }))
    expect(screen.queryByRole('listitem')).toBeNull()
  })

  it('opens the file picker from the keyboard', async () => {
    setup()
    const input = screen.getByLabelText('選擇檔案') as HTMLInputElement
    const click = vi.spyOn(input, 'click')
    screen.getByRole('button', { name: /拖曳檔案到這裡/ }).focus()
    await userEvent.keyboard('{Enter}')
    expect(click).toHaveBeenCalled()
  })

  it('accepts the supported types and allows several files in the picker', () => {
    setup()
    const input = screen.getByLabelText('選擇檔案')
    expect(input).toHaveAttribute('accept', '.pdf,.txt,.md,.png,.jpg,.jpeg,.webp')
    expect(input).toHaveAttribute('multiple')
  })

  it('stops the browser from opening a file dropped next to the zone', () => {
    setup()
    const event = createEvent.drop(window)
    fireEvent(window, event)
    expect(event.defaultPrevented).toBe(true)
  })
})
