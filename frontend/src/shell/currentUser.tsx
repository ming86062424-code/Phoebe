import { createContext, useContext, type ReactNode } from 'react'
import type { User } from '../api/types'

const CurrentUserContext = createContext<User | null>(null)

export function CurrentUserProvider({
  user,
  children,
}: {
  user: User
  children: ReactNode
}) {
  return <CurrentUserContext.Provider value={user}>{children}</CurrentUserContext.Provider>
}

export function useCurrentUser(): User {
  const user = useContext(CurrentUserContext)
  if (!user) throw new Error('useCurrentUser must be used inside CurrentUserProvider')
  return user
}
