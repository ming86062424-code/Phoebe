export interface User {
  id: number
  username: string
}

export interface Conversation {
  id: number
  title: string
  kb_id: number | null
  created_at: string
  updated_at: string
}

export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[] | null
  created_at: string
}

export interface ConversationDetail extends Conversation {
  messages: Message[]
}

export interface Source {
  document_id: number
  filename: string
  page: number
  score: number
  text: string
}

export interface KnowledgeBase {
  id: number
  name: string
  description: string
  document_count?: number
  created_at: string
  updated_at: string
}

export type DocumentStatus = 'queued' | 'parsing' | 'indexing' | 'ready' | 'failed'

export interface KbDocument {
  id: number
  kb_id: number
  filename: string
  parser: string
  status: DocumentStatus
  error: string | null
  progress_done: number
  progress_total: number
  chunk_count: number
  size_bytes: number
  created_at: string
  updated_at: string
}

export interface ParserInfo {
  id: string
  label: string
  available: boolean
  reason: string
}

export type StreamEvent =
  | { type: 'sources'; sources: Source[] }
  | { type: 'thinking'; text: string }
  | { type: 'content'; text: string }
  | { type: 'done'; message_id: number }
  | { type: 'error'; detail: string }
