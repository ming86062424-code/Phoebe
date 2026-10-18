import { render } from '@testing-library/react'
import { useEffect } from 'react'
import { describe, expect, it } from 'vitest'
import { NavGuardProvider, useNavGuard } from './navGuard'

function Blocker({ reason }: { reason: string | null }) {
  const { setBlocked } = useNavGuard()
  useEffect(() => {
    setBlocked(reason)
  }, [reason, setBlocked])
  return null
}

function fireBeforeUnload(): Event {
  const event = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(event)
  return event
}

describe('navGuard beforeunload (#3)', () => {
  it('asks before reloading or closing while blocked', () => {
    const { rerender, unmount } = render(
      <NavGuardProvider>
        <Blocker reason="回覆產生中" />
      </NavGuardProvider>,
    )
    expect(fireBeforeUnload().defaultPrevented).toBe(true)
    rerender(
      <NavGuardProvider>
        <Blocker reason={null} />
      </NavGuardProvider>,
    )
    expect(fireBeforeUnload().defaultPrevented).toBe(false)
    unmount()
  })
})
