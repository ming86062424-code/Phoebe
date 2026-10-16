import { useCallback, useState } from 'react'

function read(key: string, initial: boolean): boolean {
  try {
    const raw = localStorage.getItem(key)
    if (raw === 'true') return true
    if (raw === 'false') return false
  } catch {
    // localStorage unavailable
  }
  return initial
}

export function usePersistentState(
  key: string,
  initial: boolean,
): [boolean, (v: boolean) => void] {
  const [value, setValue] = useState(() => read(key, initial))
  const set = useCallback(
    (v: boolean) => {
      setValue(v)
      try {
        localStorage.setItem(key, String(v))
      } catch {
        // ignore write failures
      }
    },
    [key],
  )
  return [value, set]
}
