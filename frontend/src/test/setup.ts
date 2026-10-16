import '@testing-library/jest-dom/vitest'
import { ReadableStream as NodeReadableStream } from 'node:stream/web'
import { TextDecoder as NodeTextDecoder, TextEncoder as NodeTextEncoder } from 'node:util'

// jsdom may lack these; fall back to Node's implementations.
if (typeof globalThis.ReadableStream === 'undefined') {
  Object.assign(globalThis, { ReadableStream: NodeReadableStream })
}
if (typeof globalThis.TextEncoder === 'undefined') {
  Object.assign(globalThis, { TextEncoder: NodeTextEncoder })
}
if (typeof globalThis.TextDecoder === 'undefined') {
  Object.assign(globalThis, { TextDecoder: NodeTextDecoder })
}
