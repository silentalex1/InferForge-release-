import { FormEvent, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { CheckCircle2, TerminalSquare } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import AuthShell from '../components/AuthShell'
import AuthField from '../components/AuthField'

export default function Reset() {
  const { username = '' } = useParams<{ username: string }>()
  const [params] = useSearchParams()
  const code = (params.get('code') || '').trim().toUpperCase()
  const { resetPassword } = useAuth()

  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!username) {
      setError('Missing account name in the reset link.')
      return
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.')
      return
    }
    if (password !== confirm) {
      setError('Passwords do not match.')
      return
    }

    setBusy(true)
    const res = await resetPassword(username, password, code)
    setBusy(false)

    if (!res.ok) {
      setError(res.error || 'Could not reset the password.')
      return
    }
    setDone(true)
  }

  return (
    <AuthShell
      title={`Welcome ${username}`}
      subtitle="Set a new password for your InferForge account."
      footer={
        <>
          Remembered it?{' '}
          <Link to="/login" className="font-semibold text-amber-300 transition hover:text-amber-200">
            Back to login
          </Link>
        </>
      }
    >
      {!code && !done ? (
        <div className="rounded-xl border border-amber-400/25 bg-amber-400/10 px-5 py-6 text-center">
          <TerminalSquare className="mx-auto h-6 w-6 text-amber-300" />
          <p className="mt-3 text-[15px] font-semibold text-amber-100">This link needs a reset code</p>
          <p className="mt-1.5 text-[13px] leading-relaxed text-amber-200/70">
            Run this in the terminal that is connected to {username}. It opens this page with a one-time code that
            stays valid for 10 minutes.
          </p>
          <code className="mt-4 inline-block rounded-xl border border-white/10 bg-black/30 px-4 py-2.5 font-mono text-[13px] text-white/80">
            forge account reset
          </code>
        </div>
      ) : done ? (
        <div className="rounded-xl border border-emerald-400/25 bg-emerald-400/10 px-5 py-6 text-center">
          <CheckCircle2 className="mx-auto h-6 w-6 text-emerald-300" />
          <p className="mt-3 text-[15px] font-semibold text-emerald-100">Password updated</p>
          <p className="mt-1.5 text-[13px] leading-relaxed text-emerald-200/70">
            {username} can now log in with the new password.
          </p>
          <Link
            to="/login"
            className="mt-5 inline-block rounded-xl bg-amber-400 px-6 py-3 text-[14px] font-semibold text-[#0e1b3d] transition hover:bg-amber-300"
          >
            Go to login
          </Link>
        </div>
      ) : (
        <form onSubmit={submit} className="space-y-5" noValidate>
          <AuthField
            id="new-password"
            label="Enter new password"
            type="password"
            value={password}
            placeholder="New password"
            autoComplete="new-password"
            onChange={setPassword}
          />
          <AuthField
            id="confirm-password"
            label="Enter new password again"
            type="password"
            value={confirm}
            placeholder="Repeat new password"
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
            {busy ? 'Resetting...' : 'Reset password'}
          </button>
        </form>
      )}
    </AuthShell>
  )
}
