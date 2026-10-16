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

  it('shows a muted placeholder for an empty finished assistant reply', () => {
    render(<MessageBubble item={base} />)
    expect(screen.getByText('（沒有回答內容）')).toBeInTheDocument()
  })

  it('does not show the placeholder while streaming', () => {
    render(<MessageBubble item={{ ...base, status: 'streaming' }} />)
    expect(screen.queryByText('（沒有回答內容）')).toBeNull()
  })
})
