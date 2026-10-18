export interface User {
  id: number
  username: string
}

export interface Conversation {
  id: number
  title: string
  created_at: string
  updated_at: string
}

export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export interface ConversationDetail extends Conversation {
  messages: Message[]
}

export type StreamEvent =
  | { type: 'thinking'; text: string }
  | { type: 'content'; text: string }
  | { type: 'done'; message_id: number }
  | { type: 'error'; detail: string }
