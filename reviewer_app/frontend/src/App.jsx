import { useState, useEffect, lazy, Suspense } from 'react'
import { Routes, Route, useNavigate, useLocation, Navigate } from 'react-router'
import './App.css'
import Header from './components/Header'
import DocumentsPage from './pages/DocumentsPage'
import DocumentDetailPage from './pages/DocumentDetailPage'
import PhysicianDashboard from './pages/PhysicianDashboard'
// The chat pulls a heavy chunk (@ag-ui/client + shiki ≈ 2 MB). Lazy-load it so
// the bundle only ships when someone actually opens /chat.
const ChatPage = lazy(() => import('./pages/ChatPage'))
import SettingsPage from './pages/SettingsPage'
import AdminPage from './pages/AdminPage'
import { checkHealth, getCurrentUser } from './api/lakeRcmApi'
import { AutoVerdictThresholdProvider } from './components/ExtractionComparison'

function App() {
  const [, setHealth] = useState(null)
  const [currentUser, setCurrentUser] = useState(null)
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    const check = async () => {
      try {
        const data = await checkHealth()
        setHealth(data)
      } catch (err) {
        console.error('Health check failed:', err)
      }
    }
    const fetchUser = async () => {
      try {
        const data = await getCurrentUser()
        setCurrentUser(data)
      } catch (err) {
        console.error('Failed to fetch current user:', err)
      }
    }
    check()
    fetchUser()
    const interval = setInterval(check, 30000)
    return () => clearInterval(interval)
  }, [])

  const getActiveTab = () => {
    const path = location.pathname
    if (path.startsWith('/chat')) return 'chat'
    if (path.startsWith('/overview')) return 'overview'
    if (path.startsWith('/settings') || path.startsWith('/admin')) return null
    return 'review'
  }

  const handleNavigation = (tab) => {
    switch (tab) {
      case 'review':
        navigate('/review')
        break
      case 'chat':
        navigate('/chat')
        break
      case 'overview':
        navigate('/overview')
        break
      default:
        navigate('/overview')
    }
  }

  return (
    <AutoVerdictThresholdProvider value={currentUser?.auto_verdict_threshold}>
    <div className="app">
      <Header
        activeTab={getActiveTab()}
        onNavigate={handleNavigation}
        currentUser={currentUser}
      />

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<PhysicianDashboard />} />
          <Route path="/review" element={<DocumentsPage />} />
          <Route path="/review/documents/:documentId" element={<DocumentDetailPage currentUser={currentUser} />} />
          <Route
            path="/chat"
            element={
              <Suspense fallback={<div style={{ padding: '2rem' }}>Loading assistant…</div>}>
                <ChatPage currentUser={currentUser} />
              </Suspense>
            }
          />
          <Route path="/chat-v2" element={<Navigate to="/chat" replace />} />
          <Route path="/settings" element={<SettingsPage currentUser={currentUser} />} />
          <Route path="/settings/:section" element={<SettingsPage currentUser={currentUser} />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="/admin/documents/:documentId" element={<Navigate to="/review" replace />} />
          <Route path="/physician" element={<Navigate to="/overview" replace />} />
        </Routes>
      </main>
    </div>
    </AutoVerdictThresholdProvider>
  )
}

export default App
