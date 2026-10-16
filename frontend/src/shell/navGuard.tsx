import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import ConfirmDialog from './ConfirmDialog'

interface NavGuard {
  setBlocked: (reason: string | null) => void
  confirmLeave: () => Promise<boolean>
}

const NavGuardContext = createContext<NavGuard | null>(null)

export function NavGuardProvider({ children }: { children: ReactNode }) {
  const blockedRef = useRef<string | null>(null)
  const [pending, setPending] = useState<{
    message: string
    resolve: (ok: boolean) => void
  } | null>(null)

  const setBlocked = useCallback((reason: string | null) => {
    blockedRef.current = reason
  }, [])

  const confirmLeave = useCallback(() => {
    const reason = blockedRef.current
    if (reason === null) return Promise.resolve(true)
    return new Promise<boolean>((resolve) => {
      setPending({ message: reason, resolve })
    })
  }, [])

  const value = useMemo(() => ({ setBlocked, confirmLeave }), [setBlocked, confirmLeave])

  const answer = (ok: boolean) => {
    pending?.resolve(ok)
    setPending(null)
  }

  return (
    <NavGuardContext.Provider value={value}>
      {children}
      {pending && (
        <ConfirmDialog
          message={pending.message}
          onConfirm={() => answer(true)}
          onCancel={() => answer(false)}
        />
      )}
    </NavGuardContext.Provider>
  )
}

export function useNavGuard(): NavGuard {
  const ctx = useContext(NavGuardContext)
  if (!ctx) throw new Error('useNavGuard must be used inside NavGuardProvider')
  return ctx
}
