import { FormEvent, useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import AuthShell from '../components/AuthShell'
import AuthField from '../components/AuthField'

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export default function CreateAccount() {
  const { user, register } = useAuth()
  const navigate = useNavigate()

  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to="/account" replace />

  const submit = (e: FormEvent) => {
    e.preventDefault()
    setError(null)

    const name = username.trim()
    if (name.length < 3) {
      setError('Username needs at least 3 characters.')
      return
    }
    if (!EMAIL.test(email.trim())) {
      setError('Enter a valid email address.')
      return
    }
    if (password.length < 8) {
      setError('Password needs at least 8 characters.')
      return
    }
    if (password !== confirm) {
      setError('Passwords do not match.')
      return
    }

    setBusy(true)
    const res = register(name, email.trim(), password)
    setBusy(false)

    if (!res.ok) {
      setError(res.error || 'Could not create the account.')
      return
    }
    navigate('/account', { replace: true })
  }

  return (
    <AuthShell
      title="Create your account"
      subtitle="Free, and you keep running everything on your own hardware."
      footer={
        <>
          Already have one?{' '}
          <Link to="/login" className="font-semibold text-amber-300 transition hover:text-amber-200">
            Login
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
          id="email"
          label="Email"
          type="email"
          value={email}
          placeholder="you@example.com"
          autoComplete="email"
          onChange={setEmail}
        />
        <AuthField
          id="password"
          label="Password"
          type="password"
          value={password}
          placeholder="At least 8 characters"
          autoComplete="new-password"
          onChange={setPassword}
        />
        <AuthField
          id="confirm"
          label="Confirm password"
          type="password"
          value={confirm}
          placeholder="Repeat your password"
          autoComplete="new-password"
          onChange={setConfirm}
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
          {busy ? 'Creating account...' : 'Create account'}
        </button>
      </form>
    </AuthShell>
  )
}
