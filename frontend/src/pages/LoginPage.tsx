import { useEffect, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router'
import { api } from '../api/client'
import styles from './LoginPage.module.css'

export default function LoginPage() {
  const navigate = useNavigate()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  // sign-up is offered only when the server allows it (ALLOW_REGISTRATION)
  const [canRegister, setCanRegister] = useState(false)

  useEffect(() => {
    let active = true
    api
      .authOptions()
      .then((o) => {
        if (active) setCanRegister(o.allow_registration)
      })
      .catch(() => {})
    return () => {
      active = false
    }
  }, [])

  const isLogin = mode === 'login'

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      if (!isLogin) await api.register(username, password)
      await api.login(username, password)
      navigate('/chat')
    } catch (err) {
      setError(err instanceof Error ? err.message : '發生錯誤')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={styles.page}>
      <form className={styles.card} onSubmit={(e) => void submit(e)}>
        <h1 className={styles.title}>Phoebe</h1>
        <p className={styles.subtitle}>{isLogin ? '登入你的帳號' : '建立新帳號'}</p>
        <label className={styles.field}>
          <span>帳號</span>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
          />
        </label>
        <label className={styles.field}>
          <span>密碼</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={isLogin ? 'current-password' : 'new-password'}
            required
          />
        </label>
        {error && (
          <p role="alert" className={styles.error}>
            {error}
          </p>
        )}
        <button type="submit" className={styles.submit} disabled={busy}>
          {isLogin ? '登入' : '註冊'}
        </button>
        {canRegister ? (
          <button
            type="button"
            className={styles.switch}
            onClick={() => {
              setMode(isLogin ? 'register' : 'login')
              setError('')
            }}
          >
            {isLogin ? '還沒有帳號？註冊' : '已有帳號？登入'}
          </button>
        ) : (
          <p className={styles.hint}>還沒有帳號？請向管理員取得</p>
        )}
      </form>
    </div>
  )
}
