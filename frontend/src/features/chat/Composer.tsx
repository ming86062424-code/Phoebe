import { useState, type KeyboardEvent } from 'react'
import { SendHorizontal } from 'lucide-react'
import { MAX_MESSAGE_LENGTH } from './chatState'
import styles from './Composer.module.css'

interface Props {
  disabled: boolean
  onSend(text: string): void
  /** put this text back into the input (a new `n` restores again) */
  restore?: { text: string; n: number }
  error?: string | null
}

export default function Composer({ disabled, onSend, restore, error }: Props) {
  const [text, setText] = useState('')
  const [restoredN, setRestoredN] = useState(0)
  if (restore && restore.n !== restoredN) {
    setRestoredN(restore.n)
    setText(restore.text)
  }

  const trimmed = text.trim()
  const tooLong = Array.from(trimmed).length > MAX_MESSAGE_LENGTH

  const submit = () => {
    if (disabled || trimmed === '' || tooLong) return
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
      {(tooLong || error) && (
        <p role="alert" className={styles.hint}>
          {tooLong ? `訊息最多 ${MAX_MESSAGE_LENGTH} 字` : error}
        </p>
      )}
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
        disabled={disabled || trimmed === '' || tooLong}
        onClick={submit}
      >
        <SendHorizontal size={18} />
      </button>
    </div>
  )
}
