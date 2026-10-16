import type { StreamEvent } from './types'

export async function* readEvents(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<StreamEvent> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).trim()
        buffer = buffer.slice(idx + 1)
        if (line) yield JSON.parse(line) as StreamEvent
      }
    }
    buffer += decoder.decode()
    const last = buffer.trim()
    if (last) yield JSON.parse(last) as StreamEvent
  } finally {
    reader.releaseLock()
  }
}
