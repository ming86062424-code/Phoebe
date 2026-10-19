export interface User {
  id: number
  username: string
  is_admin: boolean
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

export interface SourceImage {
  name: string
  width: number
  height: number
}

export interface Source {
  document_id: number
  filename: string
  page: number // 0 for text files
  score: number // RRF
  vector_score?: number | null
  keyword_score?: number | null
  text: string // the whole chunk
  // missing on sources saved before images were indexed
  kind?: 'text' | 'page_image'
  images?: SourceImage[]
  page_image?: SourceImage | null
}

export interface KnowledgeBase {
  id: number
  name: string
  description: string
  document_count?: number
  created_at: string
  updated_at: string
}

export type DocumentStatus = 'pending' | 'queued' | 'parsing' | 'indexing' | 'ready' | 'failed'

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

export type LlmProvider = 'ollama' | 'gemini'

export interface ProviderStatus {
  available: boolean
  reason: string
}

export interface LlmSettings {
  llm_provider: LlmProvider
  ollama_model: string
  gemini_model: string
  system_prompt: string
  temperature: number | null
  providers: Record<LlmProvider, ProviderStatus>
}

export type LlmSettingsPatch = Partial<Omit<LlmSettings, 'providers'>>
