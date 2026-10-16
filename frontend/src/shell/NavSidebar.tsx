import { LogOut, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { NavLink, useNavigate } from 'react-router'
import { api } from '../api/client'
import { features } from '../features/registry'
import { useCurrentUser } from './currentUser'
import { useNavGuard } from './navGuard'
import { usePersistentState } from './usePersistentState'
import styles from './NavSidebar.module.css'

export default function NavSidebar() {
  const user = useCurrentUser()
  const navigate = useNavigate()
  const { confirmLeave } = useNavGuard()
  const [collapsed, setCollapsed] = usePersistentState('phoebe.nav.collapsed', false)

  const go = async (path: string) => {
    if (await confirmLeave()) navigate(path)
  }

  const logout = async () => {
    if (!(await confirmLeave())) return
    try {
      await api.logout()
    } catch {
      // leave to the login page regardless
    }
    navigate('/login')
  }

  const ToggleIcon = collapsed ? PanelLeftOpen : PanelLeftClose
  const toggleLabel = collapsed ? '展開選單' : '收合選單'

  return (
    <nav
      className={`${styles.nav} ${collapsed ? styles.collapsed : ''}`}
      aria-label="功能選單"
    >
      <button
        type="button"
        className={styles.item}
        aria-label={toggleLabel}
        title={toggleLabel}
        onClick={() => setCollapsed(!collapsed)}
      >
        <ToggleIcon size={20} aria-hidden />
      </button>

      <div className={styles.links}>
        {features.map((f) => (
          <NavLink
            key={f.id}
            to={f.path}
            className={({ isActive }) =>
              `${styles.item} ${isActive ? styles.active : ''}`
            }
            aria-label={collapsed ? f.label : undefined}
            title={collapsed ? f.label : undefined}
            onClick={(e) => {
              e.preventDefault()
              void go(f.path)
            }}
          >
            <f.icon size={20} aria-hidden />
            {!collapsed && <span className={styles.label}>{f.label}</span>}
          </NavLink>
        ))}
      </div>

      <div className={styles.footer}>
        {!collapsed && (
          <span className={styles.username} title={user.username}>
            {user.username}
          </span>
        )}
        <button
          type="button"
          className={styles.item}
          aria-label="登出"
          title="登出"
          onClick={() => void logout()}
        >
          <LogOut size={20} aria-hidden />
          {!collapsed && <span className={styles.label}>登出</span>}
        </button>
      </div>
    </nav>
  )
}
