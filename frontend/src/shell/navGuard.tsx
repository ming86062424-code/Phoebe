import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import ConfirmDialog from './ConfirmDialog'

interface NavGuard {
  /** Block leaving with reason; onLeave runs when the user confirms leaving anyway. */
  setBlocked: (reason: string | null, onLeave?: () => void) => void
  confirmLeave: () => Promise<boolean>
}

const NavGuardContext = createContext<NavGuard | null>(null)

export function NavGuardProvider({ children }: { children: ReactNode }) {
  const blockedRef = useRef<string | null>(null)
  const onLeaveRef = useRef<(() => void) | null>(null)
  const [pending, setPending] = useState<{
    message: string
    resolve: (ok: boolean) => void
  } | null>(null)

  const setBlocked = useCallback((reason: string | null, onLeave?: () => void) => {
    blockedRef.current = reason
    onLeaveRef.current = reason === null ? null : (onLeave ?? null)
  }, [])

  // reload and closing the tab can't show our dialog; let the browser ask instead
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (blockedRef.current === null) return
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [])

  const confirmLeave = useCallback(() => {
    const reason = blockedRef.current
    if (reason === null) return Promise.resolve(true)
    return new Promise<boolean>((resolve) => {
      setPending({
        message: reason,
        resolve: (ok) => {
          if (ok) onLeaveRef.current?.()
          resolve(ok)
        },
      })
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
