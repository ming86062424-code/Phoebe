import { describe, expect, it } from 'vitest'
import { readEvents } from './stream'
import type { StreamEvent } from './types'

function streamOf(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(c)
      controller.close()
    },
  })
}

const enc = new TextEncoder()

async function collect(body: ReadableStream<Uint8Array>): Promise<StreamEvent[]> {
  const out: StreamEvent[] = []
  for await (const e of readEvents(body)) out.push(e)
  return out
}

describe('readEvents', () => {
  it('parses events split across chunks', async () => {
    const body = streamOf([enc.encode('{"type":"con'), enc.encode('tent","text":"你"}\n')])
    expect(await collect(body)).toEqual([{ type: 'content', text: '你' }])
  })

  it('handles a multibyte char split across chunks', async () => {
    const bytes = enc.encode('{"type":"content","text":"柴犬"}\n')
    // split after the first byte of 犬 (E7 8A AC)
    const idx = enc.encode('{"type":"content","text":"柴').length + 1
    const body = streamOf([bytes.slice(0, idx), bytes.slice(idx)])
    expect(await collect(body)).toEqual([{ type: 'content', text: '柴犬' }])
  })

  it('parses a final line without trailing newline', async () => {
    const body = streamOf([enc.encode('{"type":"done","message_id":7}')])
    expect(await collect(body)).toEqual([{ type: 'done', message_id: 7 }])
  })

  it('skips blank lines', async () => {
    const body = streamOf([
      enc.encode('\n{"type":"thinking","text":"a"}\n\n   \n{"type":"done","message_id":1}\n'),
    ])
    expect(await collect(body)).toEqual([
      { type: 'thinking', text: 'a' },
      { type: 'done', message_id: 1 },
    ])
  })
})
