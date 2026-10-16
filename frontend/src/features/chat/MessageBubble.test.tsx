import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import MessageBubble from './MessageBubble'
import type { ChatItem } from './chatState'

const base: ChatItem = {
  key: 'a',
  role: 'assistant',
  content: '',
  thinking: '',
  status: 'done',
}

describe('MessageBubble', () => {
  it('renders message text literally', () => {
    const { container } = render(<MessageBubble item={{ ...base, content: '<b>x</b>' }} />)
    expect(screen.getByText('<b>x</b>')).toBeInTheDocument()
    expect(container.querySelector('b')).toBeNull()
  })

  it('shows the error and the interrupted marker', () => {
    render(
      <MessageBubble
        item={{ ...base, content: '你', status: 'error', error: '模型壞了', interrupted: true }}
      />,
    )
    expect(screen.getByText(/你.*（回覆中斷）/)).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('模型壞了')
  })
})
