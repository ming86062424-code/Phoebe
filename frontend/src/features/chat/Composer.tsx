import { useState, type KeyboardEvent } from 'react'
import { SendHorizontal } from 'lucide-react'
import styles from './Composer.module.css'

interface Props {
  disabled: boolean
  onSend(text: string): void
}

export default function Composer({ disabled, onSend }: Props) {
  const [text, setText] = useState('')

  const submit = () => {
    if (disabled || text.trim() === '') return
    onSend(text)
    setText('')
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== 'Enter' || e.shiftKey || e.nativeEvent.isComposing) return
    e.preventDefault()
    submit()
  }

  return (
    <div className={styles.composer}>
      <textarea
        className={styles.input}
        aria-label="訊息"
        placeholder="輸入訊息，Enter 送出，Shift+Enter 換行"
        rows={2}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
      />
      <button
        type="button"
        className={styles.send}
        aria-label="送出"
        disabled={disabled || text.trim() === ''}
        onClick={submit}
      >
        <SendHorizontal size={18} />
      </button>
    </div>
  )
}
