import { FormEvent, useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import AuthShell from '../components/AuthShell'
import AuthField from '../components/AuthField'

export default function Login() {
  const { user, login } = useAuth()
  const navigate = useNavigate()

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to="/account" replace />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)

    const name = username.trim()
    if (!name) {
      setError('Enter your username.')
      return
    }
    if (!password) {
      setError('Enter your password.')
      return
    }

    setBusy(true)
    const res = await login(name, password)
    setBusy(false)

    if (!res.ok) {
      setError(res.error || 'Invalid username or password.')
      return
    }
    navigate('/account', { replace: true })
  }

  return (
    <AuthShell
      title="Welcome back"
      subtitle="Login to manage your published models and embed keys."
      footer={
        <>
          No account yet?{' '}
          <Link to="/create-account" className="font-semibold text-amber-300 transition hover:text-amber-200">
            Create one
          </Link>
        </>
      }
    >
      <form onSubmit={submit} className="space-y-5" noValidate>
        <AuthField
          id="username"
          label="Username"
          type="text"
          value={username}
          placeholder="yourname"
          autoComplete="username"
          onChange={setUsername}
        />
        <AuthField
          id="password"
          label="Password"
          type="password"
          value={password}
          placeholder="Your password"
          autoComplete="current-password"
          onChange={setPassword}
        />

        {error && (
          <p className="rounded-xl border border-rose-400/25 bg-rose-400/10 px-4 py-3 text-[13px] text-rose-200">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="w-full rounded-xl bg-amber-400 px-6 py-3.5 text-[15px] font-semibold text-[#0e1b3d] transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {busy ? 'Logging in...' : 'Login'}
        </button>
      </form>
    </AuthShell>
  )
}
