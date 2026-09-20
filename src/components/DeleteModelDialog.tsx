import { FormEvent, useEffect, useState } from 'react'
import { AlertTriangle, Loader2, X } from 'lucide-react'

type Props = {
  model: string
  slug: string
  session?: string
  onClose: () => void
  onDeleted: () => void
}

export default function DeleteModelDialog({ model, slug, session, onClose, onDeleted }: Props) {
  const [typed, setTyped] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose, busy])

  const matches = typed.trim().toLowerCase() === model.toLowerCase() || typed.trim().toLowerCase() === slug

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    if (!matches) {
      setError('That does not match the model name.')
      return
    }
    if (!session) {
      setError('Your session expired. Log out and log in again, then retry.')
      return
    }
    setBusy(true)
    try {
      const res = await fetch('/api/sdk/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Forge-Session': session },
        body: JSON.stringify({ slug, confirm: typed.trim() }),
      })
      const body: any = await res.json().catch(() => ({}))
      if (!res.ok || !body?.ok) {
        setError(body?.message || body?.error || 'Could not delete this model.')
        setBusy(false)
        return
      }
      onDeleted()
    } catch {
      setError('Could not reach inferforge.org. Check your connection and try again.')
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 px-6 backdrop-blur-sm">
      <div className="w-full max-w-[460px] overflow-hidden rounded-2xl border border-rose-400/30 bg-[#101e42] shadow-2xl shadow-black/60">
        <div className="flex items-start justify-between gap-4 border-b border-rose-400/20 bg-rose-400/[0.08] px-6 py-4">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="h-5 w-5 text-rose-300" />
            <p className="text-[13px] font-bold uppercase tracking-widest text-rose-200">Warning</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            aria-label="Close"
            className="rounded-lg p-1 text-white/40 transition hover:bg-white/10 hover:text-white disabled:opacity-40"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <form onSubmit={submit} className="space-y-4 px-6 py-6" noValidate>
          <div>
            <p className="text-[15px] font-semibold text-white">
              Type the AI model name that you want to delete
            </p>
            <p className="mt-1.5 text-[13px] leading-relaxed text-rose-200/70">
              Once you delete, you cannot undo this. The hosted endpoint on inferforge.org, the SDK
              file, the chatroom on hyperneural.cfd and all request history for{' '}
              <span className="font-mono text-white/80">{model}</span> stop working immediately.
            </p>
          </div>

          <input
            autoFocus
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder="enter AI model name here"
            autoComplete="off"
            className="w-full rounded-xl border border-white/12 bg-black/30 px-4 py-3 font-mono text-[14px] text-white placeholder:font-sans placeholder:text-white/25 outline-none transition focus:border-rose-400/50 focus:ring-2 focus:ring-rose-400/15"
          />

          {error && (
            <p className="rounded-xl border border-rose-400/25 bg-rose-400/10 px-4 py-3 text-[13px] text-rose-200">
              {error}
            </p>
          )}

          <div className="flex gap-3 pt-1">
            <button
              type="button"
              onClick={onClose}
              disabled={busy}
              className="flex-1 rounded-xl border border-white/15 px-5 py-3 text-[14px] font-semibold text-white/80 transition hover:border-white/30 hover:bg-white/5 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={busy || !matches}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-rose-500 px-5 py-3 text-[14px] font-bold text-white transition hover:bg-rose-400 disabled:cursor-not-allowed disabled:bg-rose-500/40"
            >
              {busy && <Loader2 className="h-4 w-4 animate-spin" />}
              Delete
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
