import type { Conversation, ConversationDetail, User } from './types'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export function detailText(data: unknown, status: number): string {
  const detail = (data as { detail?: unknown } | null | undefined)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (d as { msg?: unknown } | null)?.msg)
      .filter((m): m is string => typeof m === 'string')
      .map((m) => m.replace(/^Value error, /, ''))
      .join('\n')
  }
  return `發生錯誤（${status}）`
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(path, init)
  if (res.status === 401) {
    window.location.hash = '#/login'
    throw new ApiError(401, await errorMessage(res))
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res))
  return res
}

async function errorMessage(res: Response): Promise<string> {
  let data: unknown = null
  try {
    data = await res.json()
  } catch {
    // body is not JSON
  }
  return detailText(data, res.status)
}

function jsonInit(method: string, body?: unknown): RequestInit {
  return body === undefined
    ? { method }
    : {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await request(path, init)
  return (await res.json()) as T
}

async function empty(path: string, init?: RequestInit): Promise<void> {
  await request(path, init)
}

export const api = {
  me: () => json<User>('/api/me'),
  login: (username: string, password: string) =>
    empty('/api/auth/login', jsonInit('POST', { username, password })),
  register: (username: string, password: string) =>
    empty('/api/auth/register', jsonInit('POST', { username, password })),
  logout: () => empty('/api/auth/logout', jsonInit('POST')),
  listConversations: () => json<Conversation[]>('/api/conversations'),
  createConversation: (signal?: AbortSignal) =>
    json<Conversation>('/api/conversations', { ...jsonInit('POST'), signal }),
  getConversation: (id: number) =>
    json<ConversationDetail>(`/api/conversations/${id}`),
  deleteConversation: (id: number) =>
    empty(`/api/conversations/${id}`, jsonInit('DELETE')),
  sendMessage: (id: number, content: string, signal?: AbortSignal) =>
    request(`/api/conversations/${id}/messages`, { ...jsonInit('POST', { content }), signal }),
  regenerate: (id: number, signal?: AbortSignal) =>
    request(`/api/conversations/${id}/regenerate`, { ...jsonInit('POST'), signal }),
}
