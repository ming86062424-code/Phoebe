import { MessageSquare, type LucideIcon } from 'lucide-react'
import { createElement, type ReactNode } from 'react'
import ChatPage from './chat/ChatPage'

export interface Feature {
  id: string
  label: string
  icon: LucideIcon
  path: string
  element: ReactNode
}

export const features: Feature[] = [
  {
    id: 'chat',
    label: 'AI 對話',
    path: '/chat',
    icon: MessageSquare,
    element: createElement(ChatPage),
  },
]
