import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, api, detailText } from './client'

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('api client', () => {
  beforeEach(() => {
    window.location.hash = ''
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('redirects to #/login and throws on 401', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(401, { detail: 'nope' })))
    await expect(api.listConversations()).rejects.toMatchObject({
      name: 'ApiError',
      status: 401,
    })
    expect(window.location.hash).toBe('#/login')
  })

  it('detailText handles string, 422 list, and fallback', () => {
    expect(detailText({ detail: 'x' }, 400)).toBe('x')
    expect(detailText({ detail: [{ msg: 'a' }, { msg: 'b' }] }, 422)).toBe('a\nb')
    expect(detailText({}, 500)).toBe('發生錯誤（500）')
    expect(detailText({ detail: [{ msg: 'Value error, 訊息不能是空白' }] }, 422)).toBe('訊息不能是空白')
  })

  it('sendMessage returns the Response on 200 and throws ApiError with detail on 503', async () => {
    const ok = new Response('', { status: 200 })
    const fetchMock = vi.fn().mockResolvedValueOnce(ok)
    vi.stubGlobal('fetch', fetchMock)
    expect(await api.sendMessage(1, 'hi')).toBe(ok)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/conversations/1/messages')

    fetchMock.mockResolvedValueOnce(jsonResponse(503, { detail: '無法連線到 Ollama' }))
    const err = await api.sendMessage(1, 'hi').catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(503)
    expect(err.message).toBe('無法連線到 Ollama')
  })

  it('returns parsed JSON and handles 204', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { id: 1, username: 'a' }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
    vi.stubGlobal('fetch', fetchMock)
    expect(await api.me()).toEqual({ id: 1, username: 'a' })
    await expect(api.logout()).resolves.toBeUndefined()
  })
})

describe('errorText (#10)', () => {
  it('uses the error message or the fallback', async () => {
    const { errorText, ApiError } = await import('./client')
    expect(errorText(new ApiError(404, '找不到'), '失敗')).toBe('找不到')
    expect(errorText('boom', '失敗')).toBe('失敗')
  })
})
