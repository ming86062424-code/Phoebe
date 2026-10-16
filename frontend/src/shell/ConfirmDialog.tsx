import { useEffect, useRef } from 'react'
import styles from './ConfirmDialog.module.css'

interface Props {
  message: string
  onConfirm: () => void
  onCancel: () => void
}

export default function ConfirmDialog({ message, onConfirm, onCancel }: Props) {
  const cancelRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    cancelRef.current?.focus()
  }, [])

  return (
    <div
      className={styles.backdrop}
      onKeyDown={(e) => {
        if (e.key === 'Escape') onCancel()
      }}
    >
      <div role="alertdialog" aria-modal="true" aria-label="確認" className={styles.dialog}>
        <p className={styles.message}>{message}</p>
        <div className={styles.actions}>
          <button ref={cancelRef} type="button" className={styles.cancel} onClick={onCancel}>
            取消
          </button>
          <button type="button" className={styles.confirm} onClick={onConfirm}>
            確定
          </button>
        </div>
      </div>
    </div>
  )
}
