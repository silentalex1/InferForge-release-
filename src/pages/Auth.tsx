import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Mail, Lock, User, ShieldCheck, ArrowRight, Loader2, CheckCircle2 } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

type Mode = 'register' | 'login'

export default function Auth({ mode }: { mode: Mode }) {
  const isRegister = mode === 'register'
  const { register, login, requestCode } = useAuth()

  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [code, setCode] = useState('')
  const [sending, setSending] = useState(false)
  const [cooldown, setCooldown] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [validated, setValidated] = useState(false)

  const handleSendCode = async () => {
    setError(null)
    setSuccess(null)
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError('Enter a valid email address first.')
      return
    }
    setSending(true)
    const c = requestCode(email)
    try {
      const r = await fetch('/api/send-code', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, code: c }) })
      if (!r.ok) throw new Error()
      setSuccess(`Verification code sent to ${email}. Check your inbox. Code: ${c}`)
    } catch {
      setSuccess(`Verification code sent to ${email}. Code: ${c}`)
    }
    setCooldown(30)
    const iv = setInterval(() => {
      setCooldown(v => {
        if (v <= 1) {
          clearInterval(iv)
          return 0
        }
        return v - 1
      })
    }, 1000)
    setTimeout(() => setSending(false), 600)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    if (!username.trim()) { setError('Enter account username.'); return }
    if (!password || password.length < 6) { setError('Password must be at least 6 characters.'); return }
    if (isRegister) {
      if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) { setError('Enter a valid email.'); return }
      if (!code.trim()) { setError('Enter verification code that we will send.'); return }
      try {
        const raw = localStorage.getItem('inferforge-pending-code')
        if (!raw) { setError('Please request a verification code first.'); return }
        const pending = JSON.parse(raw)
        if (pending.email !== email.toLowerCase()) { setError('Verification code does not match this email.'); return }
        if (Date.now() > pending.expires) { setError('Verification code expired. Send a new one.'); return }
        if (pending.code !== code.trim()) { setError('Incorrect verification code.'); return }
      } catch {
        setError('Verification failed. Request a new code.')
        return
      }
      const res = register(username.trim(), email.trim(), password)
      if (!res.ok) { setError(res.error || 'Registration failed.'); return }
      localStorage.removeItem('inferforge-pending-code')
      setValidated(true)
      setSuccess('Code validated. Account created!')
      setTimeout(() => window.location.reload(), 1100)
      return
    } else {
      const res = await login(username.trim(), password)
      if (!res.ok) { setError(res.error || 'Login failed.'); return }
      setValidated(true)
      setSuccess('Logged in successfully!')
      setTimeout(() => window.location.reload(), 900)
      return
    }
  }

  return (
    <>
      {validated && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-[90] animate-[slideDown_0.35s_ease]">
          <div className="flex items-center gap-2.5 px-5 py-3 rounded-full bg-emerald-500 text-white text-sm font-semibold shadow-[0_12px_32px_rgba(16,185,129,0.35)] border border-emerald-400">
            <CheckCircle2 className="w-5 h-5" />
            {isRegister ? 'Code validated. Account created!' : 'Logged in successfully!'}
          </div>
        </div>
      )}
      <div className="min-h-[calc(100vh-64px)] flex items-center justify-center bg-[#0A0A0B] px-6 py-12">
      <div className="w-full max-w-[440px]">
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2.5">
            <span className="w-8 h-8 rounded-lg bg-white flex items-center justify-center">
              <span className="text-xs font-extrabold tracking-tighter text-black">IF</span>
            </span>
            <span className="text-[15px] font-bold tracking-tight text-white">InferForge</span>
          </Link>
          <h1 className="mt-6 text-2xl font-bold tracking-tight text-white">{isRegister ? 'Create account' : 'Login to account'}</h1>
          <p className="mt-2 text-sm text-white/45">
            {isRegister ? 'We will send a verification code to your email so make sure your email is the correct one you use.' : 'Welcome back. Enter your credentials to continue.'}
          </p>
        </div>

        <div className="rounded-2xl border border-white/[0.07] bg-white/[0.03] backdrop-blur p-6 sm:p-7 shadow-[0_16px_48px_rgba(0,0,0,0.35)]">
          <div className="flex p-1 rounded-full bg-white/[0.06] border border-white/[0.06] mb-6">
            <Link to="/register" className={`flex-1 py-2 rounded-full text-sm font-semibold text-center transition ${isRegister ? 'bg-white text-black shadow' : 'text-white/50 hover:text-white'}`}>Create account</Link>
            <Link to="/login" className={`flex-1 py-2 rounded-full text-sm font-semibold text-center transition ${!isRegister ? 'bg-white text-black shadow' : 'text-white/50 hover:text-white'}`}>Login to account</Link>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <label className="block">
              <span className="text-xs font-medium text-white/60">Enter account username</span>
              <div className="mt-1.5 relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25" />
                <input value={username} onChange={e => setUsername(e.target.value)} placeholder="Enter account username" className="w-full pl-10 pr-3 py-2.5 rounded-xl bg-white/[0.06] border border-white/[0.08] text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-[#FF7A00]/50 focus:bg-white/[0.08] transition" />
              </div>
            </label>

            {isRegister && (
              <label className="block">
                <span className="text-xs font-medium text-white/60">Enter your email</span>
                <div className="mt-1.5 relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25" />
                  <input value={email} onChange={e => setEmail(e.target.value)} placeholder="Enter your email" type="email" className="w-full pl-10 pr-3 py-2.5 rounded-xl bg-white/[0.06] border border-white/[0.08] text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-[#FF7A00]/50 focus:bg-white/[0.08] transition" />
                </div>
              </label>
            )}

            <label className="block">
              <span className="text-xs font-medium text-white/60">Enter account password</span>
              <div className="mt-1.5 relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25" />
                <input value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter account password" type="password" className="w-full pl-10 pr-3 py-2.5 rounded-xl bg-white/[0.06] border border-white/[0.08] text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-[#FF7A00]/50 focus:bg-white/[0.08] transition" />
              </div>
            </label>

            {isRegister && (
              <label className="block">
                <span className="text-xs font-medium text-white/60">Enter verification code that we will send.</span>
                <div className="mt-1.5 flex gap-2">
                  <div className="relative flex-1">
                    <ShieldCheck className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25" />
                    <input value={code} onChange={e => setCode(e.target.value)} placeholder="Enter verification code" className="w-full pl-10 pr-3 py-2.5 rounded-xl bg-white/[0.06] border border-white/[0.08] text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-[#FF7A00]/50 focus:bg-white/[0.08] transition" />
                  </div>
                  <button type="button" onClick={handleSendCode} disabled={sending || cooldown > 0} className="px-4 py-2.5 rounded-xl bg-white text-black text-xs font-bold hover:bg-white/90 transition disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap flex items-center gap-1.5">
                    {sending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
                    {cooldown > 0 ? `${cooldown}s` : 'Send verification code'}
                  </button>
                </div>
              </label>
            )}

            {error && <p className="text-xs leading-5 px-3 py-2.5 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300">{error}</p>}
            {success && <p className="text-xs leading-5 px-3 py-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">{success}</p>}

            <button type="submit" className="w-full inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-[#FF7A00] text-sm font-bold text-white hover:bg-[#ff8c1a] transition shadow-[0_8px_20px_rgba(255,122,0,0.3)]">
              {isRegister ? 'Create account' : 'Login'}
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          <p className="mt-6 text-center text-xs text-white/35">
            {isRegister ? (
              <>Already have an account? <Link to="/login" className="text-white font-semibold hover:underline underline-offset-4">Login to account</Link></>
            ) : (
              <>No account yet? <Link to="/register" className="text-white font-semibold hover:underline underline-offset-4">Create account</Link></>
            )}
          </p>
        </div>
      </div>
    </div>
    <style>{`@keyframes slideDown{from{transform:translate(-50%,-16px);opacity:0}to{transform:translate(-50%,0);opacity:1}}`}</style>
    </>
  )
}
