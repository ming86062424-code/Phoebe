import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { ApiError, api } from '../api/client'
import type { User } from '../api/types'
import { CurrentUserProvider } from './currentUser'

export default function RequireAuth({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false
    api
      .me()
      .then((u) => {
        if (!cancelled) setUser(u)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        // 401 is already redirected to #/login by the API client
        if (err instanceof ApiError && err.status === 401) return
        setError(err instanceof ApiError ? err.message : '無法連線到伺服器')
      })
    return () => {
      cancelled = true
    }
  }, [attempt])

  const retry = useCallback(() => {
    setError(null)
    setAttempt((n) => n + 1)
  }, [])

  if (error !== null) {
    return (
      <div role="alert">
        <p>{error}</p>
        <button type="button" onClick={retry}>
          重試
        </button>
      </div>
    )
  }
  if (!user) return <div role="status">載入中…</div>
  return <CurrentUserProvider user={user}>{children}</CurrentUserProvider>
}
