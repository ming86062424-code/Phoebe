import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { User } from '../api/types'
import { CurrentUserProvider } from './currentUser'

export default function RequireAuth({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .me()
      .then((u) => {
        if (!cancelled) setUser(u)
      })
      .catch(() => {
        // 401 already redirected to #/login by the API client
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (!user) return <div role="status">載入中…</div>
  return <CurrentUserProvider user={user}>{children}</CurrentUserProvider>
}
