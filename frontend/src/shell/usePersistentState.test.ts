import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { usePersistentState } from './usePersistentState'

describe('usePersistentState', () => {
  beforeEach(() => localStorage.clear())
  afterEach(() => vi.restoreAllMocks())

  it('reads and writes localStorage', () => {
    localStorage.setItem('k', 'true')
    const { result } = renderHook(() => usePersistentState('k', false))
    expect(result.current[0]).toBe(true)
    act(() => result.current[1](false))
    expect(result.current[0]).toBe(false)
    expect(localStorage.getItem('k')).toBe('false')
  })

  it('falls back to default when localStorage throws', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    const { result } = renderHook(() => usePersistentState('k', false))
    expect(result.current[0]).toBe(false)
    expect(() => act(() => result.current[1](true))).not.toThrow()
    expect(result.current[0]).toBe(true)
  })
})
