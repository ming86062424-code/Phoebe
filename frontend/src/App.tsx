import { HashRouter, Navigate, Outlet, Route, Routes } from 'react-router'
import { features, type Feature } from './features/registry'
import LoginPage from './pages/LoginPage'
import AppShell from './shell/AppShell'
import RequireAuth from './shell/RequireAuth'
import { useCurrentUser } from './shell/currentUser'

function FeatureRoute({ feature }: { feature: Feature }) {
  const user = useCurrentUser()
  if (feature.adminOnly && !user.is_admin) return <Navigate to="/chat" replace />
  return feature.element
}

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
            <Route key={f.id} path={`${f.path}/*`} element={<FeatureRoute feature={f} />} />
          ))}
        </Route>
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </HashRouter>
  )
}
