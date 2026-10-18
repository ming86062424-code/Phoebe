import { BookOpen, MessageSquare, Settings, type LucideIcon } from 'lucide-react'
import { createElement, type ReactNode } from 'react'
import ChatPage from './chat/ChatPage'
import KbPage from './kb/KbPage'
import SettingsPage from './settings/SettingsPage'

export interface Feature {
  id: string
  label: string
  icon: LucideIcon
  path: string
  element: ReactNode
  adminOnly?: boolean
}

export const features: Feature[] = [
  {
    id: 'chat',
    label: 'AI 對話',
    path: '/chat',
    icon: MessageSquare,
    element: createElement(ChatPage),
  },
  {
    id: 'kb',
    label: '知識庫',
    path: '/kb',
    icon: BookOpen,
    element: createElement(KbPage),
  },
  {
    id: 'settings',
    label: '系統設定',
    path: '/settings',
    icon: Settings,
    element: createElement(SettingsPage),
    adminOnly: true,
  },
]
