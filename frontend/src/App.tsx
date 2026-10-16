import { HashRouter, Navigate, Route, Routes } from 'react-router'
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
        {features.flatMap((f) =>
          [f.path, `${f.path}/*`].map((path) => (
            <Route
              key={`${f.id}:${path}`}
              path={path}
              element={
                <RequireAuth>
                  <AppShell>{f.element}</AppShell>
                </RequireAuth>
              }
            />
          )),
        )}
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </HashRouter>
  )
}
