import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, Navigate, useParams } from 'react-router-dom'
import { Activity, AlertTriangle, Boxes, Check, Copy, ExternalLink, RefreshCw, Zap } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

type Requests = {
  total: number
  ok: number
  failed: number
  today: number
  days: Record<string, number>
  last_request_at: string | null
  last_status: number | null
}

type Model = {
  model: string
  slug: string
  owner: string | null
  endpoint: string
  sdk_url: string
  embed: string
  key: string
  created_at: string
  updated_at: string
  status: 'online' | 'offline' | 'private'
  online: boolean
  requests: Requests
}

type Payload = {
  count: number
  totals: { requests: number; today: number; failed: number; online: number }
  checked_at: string
  models: Model[]
}

const POLL_MS = 15000

const statusStyles: Record<Model['status'], { dot: string; text: string; label: string }> = {
  online: { dot: 'bg-emerald-400', text: 'text-emerald-300', label: 'Online' },
  offline: { dot: 'bg-rose-400', text: 'text-rose-300', label: 'Offline' },
  private: { dot: 'bg-amber-400', text: 'text-amber-300', label: 'Private address' },
}

function timeAgo(iso: string | null): string {
  if (!iso) return 'never'
  const diff = Date.now() - new Date(iso).getTime()
  if (Number.isNaN(diff)) return 'never'
  const s = Math.floor(diff / 1000)
  if (s < 60) return s + 's ago'
  const m = Math.floor(s / 60)
  if (m < 60) return m + 'm ago'
  const h = Math.floor(m / 60)
  if (h < 24) return h + 'h ago'
  return Math.floor(h / 24) + 'd ago'
}

function CopyButton({ value, label }: { value: string; label: string }) {
  const [done, setDone] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setDone(true)
      setTimeout(() => setDone(false), 1800)
    } catch {
      setDone(false)
    }
  }
  return (
    <button
      type="button"
      onClick={copy}
      className="inline-flex items-center gap-1.5 rounded-lg border border-white/12 px-2.5 py-1.5 text-[12px] font-medium text-white/60 transition hover:border-white/25 hover:text-white"
    >
      {done ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
      {done ? 'Copied' : label}
    </button>
  )
}

function Sparkline({ days }: { days: Record<string, number> }) {
  const keys = Object.keys(days).sort().slice(-14)
  if (keys.length === 0) return <div className="h-9 text-[12px] text-white/25">No requests yet</div>
  const max = Math.max(...keys.map((k) => days[k]), 1)
  return (
    <div className="flex h-9 items-end gap-1">
      {keys.map((k) => (
        <div
          key={k}
          title={k + ': ' + days[k]}
          style={{ height: Math.max(3, (days[k] / max) * 36) }}
          className="w-2 rounded-sm bg-amber-400/60"
        />
      ))}
    </div>
  )
}

export default function UserDashboard() {
  const { username } = useParams<{ username: string }>()
  const { user } = useAuth()

  const [data, setData] = useState<Payload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const owner = user?.username || ''

  const load = useCallback(
    async (quiet: boolean) => {
      if (!owner) return
      if (quiet) setRefreshing(true)
      try {
        const res = await fetch('/api/sdk/models?owner=' + encodeURIComponent(owner))
        if (!res.ok) throw new Error('registry returned ' + res.status)
        const body = (await res.json()) as Payload
        setData(body)
        setError(null)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Could not reach the model registry.')
      } finally {
        setLoading(false)
        setRefreshing(false)
      }
    },
    [owner]
  )

  useEffect(() => {
    if (!owner) return
    load(false)
    const id = setInterval(() => load(true), POLL_MS)
    return () => clearInterval(id)
  }, [owner, load])

  const models = useMemo(() => data?.models ?? [], [data])

  if (!user) return <Navigate to="/login" replace />
  if (username !== user.username) {
    return <Navigate to={`/dashboard/${encodeURIComponent(user.username)}`} replace />
  }

  const totals = data?.totals ?? { requests: 0, today: 0, failed: 0, online: 0 }

  const cards = [
    { label: 'Published models', value: String(data?.count ?? 0), hint: 'via forge embedd' },
    { label: 'Online now', value: String(totals.online), hint: 'live upstream check' },
    { label: 'Requests today', value: String(totals.today), hint: 'across all models' },
    { label: 'Requests total', value: String(totals.requests), hint: totals.failed + ' failed' },
  ]

  return (
    <div className="max-w-6xl mx-auto px-6 py-12 md:py-16">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[30px] font-bold tracking-tight md:text-[36px]">Dashboard</h1>
          <p className="mt-2 text-[15px] text-white/50">
            Signed in as <span className="font-semibold text-white/80">{user.username}</span>
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[12px] text-white/35">
            {data ? 'Checked ' + timeAgo(data.checked_at) : 'Loading'}
          </span>
          <button
            type="button"
            onClick={() => load(true)}
            className="inline-flex items-center gap-2 rounded-lg border border-white/12 px-3 py-2 text-[13px] font-medium text-white/70 transition hover:border-white/25 hover:text-white"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-8 flex items-start gap-3 rounded-2xl border border-rose-400/25 bg-rose-400/10 px-5 py-4">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-300" />
          <div>
            <p className="text-[14px] font-semibold text-rose-100">Could not load your models</p>
            <p className="mt-1 text-[13px] text-rose-200/70">{error}</p>
          </div>
        </div>
      )}

      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((c) => (
          <div key={c.label} className="rounded-2xl border border-white/10 bg-white/[0.035] p-5">
            <p className="text-[12px] font-medium uppercase tracking-wide text-white/35">{c.label}</p>
            <p className="mt-3 font-mono text-[30px] font-bold tracking-tight">{loading ? '·' : c.value}</p>
            <p className="mt-1 text-[12px] text-white/35">{c.hint}</p>
          </div>
        ))}
      </div>

      <div className="mt-12 flex items-center gap-2.5">
        <Boxes className="h-5 w-5 text-amber-300/70" />
        <h2 className="text-[20px] font-semibold tracking-tight">Your embedded models</h2>
      </div>

      {!loading && models.length === 0 && !error && (
        <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.035] px-6 py-10 text-center">
          <Zap className="mx-auto h-6 w-6 text-amber-300/60" />
          <p className="mt-4 text-[16px] font-semibold">Nothing published yet</p>
          <p className="mx-auto mt-2 max-w-md text-[14px] leading-relaxed text-white/45">
            Connect the CLI with <code className="font-mono text-amber-200">forge connect</code>, then publish a
            model. It shows up here with live status and request counts.
          </p>
          <code className="mt-6 inline-block rounded-xl border border-white/10 bg-black/30 px-4 py-2.5 font-mono text-[13px] text-white/75">
            forge embedd your-model --sdk
          </code>
        </div>
      )}

      <div className="mt-6 space-y-4">
        {models.map((m) => {
          const s = statusStyles[m.status]
          return (
            <div key={m.slug} className="rounded-2xl border border-white/10 bg-white/[0.035] p-6">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2.5">
                    <span className={`h-2 w-2 shrink-0 rounded-full ${s.dot}`} />
                    <h3 className="truncate text-[18px] font-semibold tracking-tight">{m.model}</h3>
                    <span className={`text-[12px] font-medium ${s.text}`}>{s.label}</span>
                  </div>
                  <p className="mt-1.5 truncate font-mono text-[12px] text-white/35">{m.endpoint}</p>
                </div>
                <a
                  href={m.sdk_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-[13px] font-medium text-amber-300 transition hover:text-amber-200"
                >
                  Open SDK
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              </div>

              {m.status === 'private' && (
                <p className="mt-4 rounded-xl border border-amber-400/20 bg-amber-400/[0.07] px-4 py-3 text-[13px] leading-relaxed text-amber-100/80">
                  This model points at a private address, so only your own browser can reach it. Re-run{' '}
                  <code className="font-mono">forge embedd {m.model} --sdk --endpoint https://your-public-url</code> to
                  serve it publicly.
                </p>
              )}

              <div className="mt-6 grid gap-6 lg:grid-cols-[1.2fr_1fr]">
                <div>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-[11px] uppercase tracking-wide text-white/30">Total</p>
                      <p className="mt-1 font-mono text-[20px] font-bold">{m.requests.total}</p>
                    </div>
                    <div>
                      <p className="text-[11px] uppercase tracking-wide text-white/30">Today</p>
                      <p className="mt-1 font-mono text-[20px] font-bold">{m.requests.today}</p>
                    </div>
                    <div>
                      <p className="text-[11px] uppercase tracking-wide text-white/30">Failed</p>
                      <p className="mt-1 font-mono text-[20px] font-bold text-white/60">{m.requests.failed}</p>
                    </div>
                  </div>
                  <div className="mt-5">
                    <p className="text-[11px] uppercase tracking-wide text-white/30">Last 14 days</p>
                    <div className="mt-2">
                      <Sparkline days={m.requests.days} />
                    </div>
                  </div>
                  <p className="mt-4 flex items-center gap-1.5 text-[12px] text-white/35">
                    <Activity className="h-3.5 w-3.5" />
                    Last request {timeAgo(m.requests.last_request_at)}
                  </p>
                </div>

                <div>
                  <p className="text-[11px] uppercase tracking-wide text-white/30">Embed snippet</p>
                  <pre className="mt-2 overflow-x-auto rounded-xl border border-white/10 bg-black/30 px-4 py-3 font-mono text-[11px] leading-relaxed text-white/65">
                    {m.embed}
                  </pre>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <CopyButton value={m.embed} label="Copy embed" />
                    <CopyButton value={m.sdk_url} label="Copy URL" />
                    <CopyButton value={m.key} label="Copy key" />
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="mt-12 rounded-2xl border border-white/10 bg-white/[0.02] p-6">
        <h2 className="text-[16px] font-semibold tracking-tight">Publishing a model</h2>
        <ol className="mt-4 space-y-2.5 text-[14px] leading-relaxed text-white/55">
          <li>
            <span className="font-mono text-amber-200">forge connect</span> links this account to your terminal.
          </li>
          <li>
            <span className="font-mono text-amber-200">forge serve</span> keeps the model loaded and answering.
          </li>
          <li>
            <span className="font-mono text-amber-200">forge embedd &lt;model&gt; --sdk</span> publishes it and prints
            your script tag.
          </li>
        </ol>
        <Link
          to="/docs"
          className="mt-5 inline-flex items-center gap-2 text-[14px] font-semibold text-amber-300 transition hover:text-amber-200"
        >
          Read the deployment docs
          <ExternalLink className="h-3.5 w-3.5" />
        </Link>
      </div>
    </div>
  )
}
