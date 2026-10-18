import { HashRouter, Navigate, Outlet, Route, Routes } from 'react-router'
import { features } from './features/registry'
import LoginPage from './pages/LoginPage'
import AppShell from './shell/AppShell'
import RequireAuth from './shell/RequireAuth'

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route
          element={
            <RequireAuth>
              <AppShell>
                <Outlet />
              </AppShell>
            </RequireAuth>
          }
        >
          {features.map((f) => (
            <Route key={f.id} path={`${f.path}/*`} element={f.element} />
          ))}
        </Route>
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </HashRouter>
  )
}
