import { FormEvent, useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { AlertTriangle, ArrowRight, CheckCircle2, LayoutDashboard, Loader2, ShieldCheck } from 'lucide-react'
import { useAuth } from '../context/AuthContext'


type Step = 'checking' | 'signed-in' | 'need-account' | 'code' | 'confirmed' | 'error'

function localAccount(name: string): { username: string; email: string } | null {
  try {
    const all = JSON.parse(localStorage.getItem('inferforge-users') || '[]')
    const hit = all.find((u: any) => String(u.username).toLowerCase() === name.toLowerCase())
    return hit ? { username: hit.username, email: hit.email } : null
  } catch {
    return null
  }
}

export default function Account() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const { user } = useAuth()
  const verifyUsername = (params.get('verify') || '').trim()

  const [step, setStep] = useState<Step>(verifyUsername ? 'checking' : user ? 'signed-in' : 'need-account')
  const [message, setMessage] = useState<string | null>(null)
  const [code, setCode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [countdown, setCountdown] = useState(5)
  const [username, setUsername] = useState(verifyUsername || user?.username || '')
  const [email, setEmail] = useState(user?.email || '')

  useEffect(() => {
    if (!verifyUsername) {
      setStep(user ? 'signed-in' : 'need-account')
      setUsername(user?.username || '')
      setEmail(user?.email || '')
      return
    }

    let cancelled = false
    setUsername(verifyUsername)

    if (user && user.username.toLowerCase() === verifyUsername.toLowerCase()) {
      setEmail(user.email)
      setStep('code')
      return
    }

    const stored = localAccount(verifyUsername)
    if (stored) {
      setEmail(stored.email)
      setStep('code')
      return
    }

    fetch(`/api/auth/user/${encodeURIComponent(verifyUsername)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data: any) => {
        if (cancelled) return
        if (data && data.username) {
          setEmail(data.email || '')
          setStep('code')
        } else {
          setStep('need-account')
          setMessage(`No account found for "${verifyUsername}". Create one here, then run forge connect again.`)
        }
      })
      .catch(() => {
        if (cancelled) return
        setStep('need-account')
        setMessage(`Could not look up "${verifyUsername}". Check your connection and try again.`)
      })

    return () => {
      cancelled = true
    }
  }, [verifyUsername, user])

  useEffect(() => {
    if (step !== 'confirmed') return
    const iv = setInterval(() => {
      setCountdown((v) => {
        if (v <= 1) {
          clearInterval(iv)
          navigate(username ? `/dashboard/${encodeURIComponent(username)}` : '/')
          return 0
        }
        return v - 1
      })
    }, 1000)
    return () => clearInterval(iv)
  }, [step, navigate, username])

  const handleConfirm = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    if (code.trim().length !== 5) {
      setError('Paste in the 5 character code shown in your terminal.')
      return
    }
    setLoading(true)
    try {
      const res = await fetch('/api/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, code: code.trim().toUpperCase(), email, confirm: true }),
      })
      if (!res.ok) throw new Error('rejected')
      setStep('confirmed')
    } catch {
      setError('Could not confirm the code. Check the code in your terminal and try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[460px] bg-[radial-gradient(ellipse_55%_50%_at_50%_0%,rgba(251,191,36,0.1),transparent_70%)]" />

      <div className="relative mx-auto flex min-h-[calc(100vh-64px)] w-full max-w-[460px] flex-col justify-center px-6 py-16">
        <div className="text-center">
          <h1 className="text-[26px] font-bold tracking-tight">
            {step === 'signed-in' ? 'Your account' : 'Connect account'}
          </h1>
          <p className="mt-2 text-[14px] text-white/45">
            {step === 'confirmed'
              ? 'Your terminal is now connected.'
              : step === 'signed-in'
                ? 'Signed in on this browser.'
                : 'Paste in the code from your terminal to connect.'}
          </p>
        </div>

        <div className="mt-8 rounded-2xl border border-white/10 bg-white/[0.035] p-6 sm:p-7">
          {step === 'checking' && (
            <div className="flex items-center justify-center gap-3 py-8 text-[14px] text-white/55">
              <Loader2 className="h-5 w-5 animate-spin" />
              Checking account
            </div>
          )}

          {step === 'signed-in' && user && (
            <div className="space-y-5">
              <div className="rounded-xl border border-emerald-400/25 bg-emerald-400/[0.08] px-4 py-3.5">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  <p className="text-[14px] font-semibold text-emerald-100">Account verified</p>
                </div>
                <p className="mt-1.5 text-[13px] text-emerald-200/70">
                  {user.username} · {user.email}
                </p>
              </div>

              <Link
                to={`/dashboard/${encodeURIComponent(user.username)}`}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-amber-400 px-5 py-3 text-[15px] font-semibold text-[#0e1b3d] transition hover:bg-amber-300"
              >
                <LayoutDashboard className="h-4 w-4" />
                Go to dashboard
              </Link>

              <div className="rounded-xl border border-white/10 bg-black/25 px-4 py-3.5">
                <p className="text-[13px] text-white/50">To link this account to your terminal, run:</p>
                <code className="mt-2 block font-mono text-[13px] text-amber-200">forge connect</code>
                <p className="mt-2 text-[12px] leading-relaxed text-white/35">
                  That opens this page with a code box and a 5 character code in your terminal.
                </p>
              </div>
            </div>
          )}

          {step === 'need-account' && (
            <div className="space-y-4">
              <div className="flex items-start gap-3 rounded-xl border border-amber-400/25 bg-amber-400/10 p-4">
                <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
                <div>
                  <p className="text-[14px] font-medium text-amber-100">
                    {verifyUsername ? 'No account found' : 'Not signed in on this browser'}
                  </p>
                  <p className="mt-1 text-[12px] leading-5 text-amber-200/70">
                    {message ||
                      'Log in to see your account, or create one first so the terminal has something to connect to.'}
                  </p>
                </div>
              </div>
              <Link
                to="/create-account"
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-amber-400 px-5 py-3 text-[15px] font-semibold text-[#0e1b3d] transition hover:bg-amber-300"
              >
                Create account
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                to="/login"
                className="inline-flex w-full items-center justify-center rounded-xl border border-white/15 px-5 py-3 text-[15px] font-semibold text-white/85 transition hover:border-white/30 hover:bg-white/5"
              >
                Login
              </Link>
            </div>
          )}

          {step === 'code' && (
            <form onSubmit={handleConfirm} className="space-y-4" noValidate>
              <div className="text-center">
                <p className="text-[12px] text-white/45">Connecting account</p>
                <p className="text-[14px] font-semibold text-white">
                  {username}
                  {email ? ' · ' + email : ''}
                </p>
              </div>
              <label htmlFor="code" className="block">
                <span className="text-[12px] font-medium text-white/55">Enter code here</span>
                <div className="relative mt-1.5">
                  <ShieldCheck className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-white/25" />
                  <input
                    id="code"
                    value={code}
                    onChange={(e) => setCode(e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 5))}
                    placeholder="Enter code here"
                    maxLength={5}
                    className="w-full rounded-xl border border-white/12 bg-black/25 py-3 pl-10 pr-3 text-center font-mono text-[15px] tracking-[0.4em] text-white placeholder:tracking-normal placeholder:text-white/25 outline-none transition focus:border-amber-400/50 focus:ring-2 focus:ring-amber-400/15"
                  />
                </div>
              </label>
              {error && (
                <p className="rounded-xl border border-rose-400/25 bg-rose-400/10 px-4 py-3 text-[13px] text-rose-200">
                  {error}
                </p>
              )}
              <button
                type="submit"
                disabled={loading}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-amber-400 px-5 py-3 text-[15px] font-semibold text-[#0e1b3d] transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                Connect
              </button>
            </form>
          )}

          {step === 'confirmed' && (
            <div className="space-y-5 py-4 text-center">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-emerald-400/30 bg-emerald-400/15">
                <CheckCircle2 className="h-7 w-7 text-emerald-400" />
              </div>
              <div>
                <p className="text-[18px] font-bold text-emerald-300">Welcome {username}</p>
                <p className="mt-1 text-[14px] text-white/50">
                  Account connected. Taking you to your dashboard in {countdown}s.
                </p>
              </div>
              <div className="h-1 overflow-hidden rounded-full bg-white/[0.07]">
                <div
                  className="h-full bg-emerald-400 transition-all duration-1000"
                  style={{ width: `${((5 - countdown) / 5) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>

        <p className="mt-6 text-center text-[12px] text-white/30">
          Terminal command: <span className="font-mono text-white/55">forge connect</span>
        </p>
      </div>
    </div>
  )
}
