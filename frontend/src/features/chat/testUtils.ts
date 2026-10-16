import type { StreamEvent } from '../../api/types'

const encoder = new TextEncoder()

export function ndjsonResponse(events: StreamEvent[], rawTail = ''): Response {
  const text = events.map((e) => JSON.stringify(e) + '\n').join('') + rawTail
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(text))
        controller.close()
      },
    }),
    { status: 200, headers: { 'Content-Type': 'application/x-ndjson' } },
  )
}

export interface HeldStream {
  response: Response
  push(event: StreamEvent): void
  close(): void
}

export function heldResponse(): HeldStream {
  let ctrl!: ReadableStreamDefaultController<Uint8Array>
  const response = new Response(
    new ReadableStream({
      start(c) {
        ctrl = c
      },
    }),
    { status: 200, headers: { 'Content-Type': 'application/x-ndjson' } },
  )
  return {
    response,
    push: (event) => ctrl.enqueue(encoder.encode(JSON.stringify(event) + '\n')),
    close: () => ctrl.close(),
  }
}

export function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

export function conversation(
  id: number,
  messages: { role: 'user' | 'assistant'; content: string }[] = [],
) {
  return {
    id,
    title: `對話${id}`,
    kb_id: null as number | null,
    created_at: '2026-01-01T00:00:00',
    updated_at: '2026-01-01T00:00:00',
    messages: messages.map((m, i) => ({
      id: i + 1,
      created_at: '2026-01-01T00:00:00',
      ...m,
    })),
  }
}
