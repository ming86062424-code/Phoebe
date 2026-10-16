import type { ReactNode } from 'react'
import { NavGuardProvider } from './navGuard'
import NavSidebar from './NavSidebar'
import styles from './AppShell.module.css'

export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <NavGuardProvider>
      <div className={styles.page}>
        <div className={styles.card}>
          <NavSidebar />
          <main className={styles.main}>{children}</main>
        </div>
      </div>
    </NavGuardProvider>
  )
}
