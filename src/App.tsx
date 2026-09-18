import { Routes, Route, useLocation, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Download from './pages/Download'
import Models from './pages/Models'
import Docs from './pages/Docs'
import Chat from './pages/Chat'
import Pricing from './pages/Pricing'
import Login from './pages/Login'
import CreateAccount from './pages/CreateAccount'
import Account from './pages/Account'
import Reset from './pages/Reset'
import UserDashboard from './pages/UserDashboard'
import OurModels from './pages/OurModels'
import Batnight from './pages/Batnight'

export default function App() {
  const location = useLocation()
  const isChat = location.pathname === '/chat' || location.pathname === '/chatui' || location.pathname === '/v1/chatui'

  if (isChat) {
    return <Chat />
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#0e1b3d] text-white antialiased">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/download" element={<Download />} />
          <Route path="/models" element={<Models />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="/pricing" element={<Pricing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/create-account" element={<CreateAccount />} />
          <Route path="/register" element={<Navigate to="/create-account" replace />} />
          <Route path="/registar" element={<Navigate to="/create-account" replace />} />
          <Route path="/signup" element={<Navigate to="/create-account" replace />} />
          <Route path="/account" element={<Account />} />
          <Route path="/reset/:username" element={<Reset />} />
          <Route path="/dashboard/:username" element={<UserDashboard />} />
          <Route path="/our-models" element={<OurModels />} />
          <Route path="/batnight" element={<Batnight />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/chatui" element={<Chat />} />
          <Route path="/v1/chatui" element={<Chat />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}
