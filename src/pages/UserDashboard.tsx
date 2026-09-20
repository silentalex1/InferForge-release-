import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, Navigate, useLocation, useParams } from 'react-router-dom'
import {
  Activity,
  AlertTriangle,
  Boxes,
  Check,
  Copy,
  Crown,
  ExternalLink,
  KeyRound,
  LayoutDashboard,
  MessageSquare,
  Radio,
  RefreshCw,
  Server,
  Settings,
  Trash2,
  Zap,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import DeleteModelDialog from '../components/DeleteModelDialog'

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
  status: 'online' | 'hosted' | 'offline' | 'private'
  online: boolean
  upstream_private?: boolean
  hosted_model?: string
  has_persona?: boolean
  requests: Requests
}

type Payload = {
  count: number
  role: 'owner' | 'premium' | 'free'
  premium: boolean
  limit: number | null
  totals: { requests: number; today: number; failed: number; online: number }
  checked_at: string
  models: Model[]
}

const POLL_MS = 15000
const CHATROOM_BASE = 'https://hyperneural.cfd/chatroom/'

function chatroomUrl(slug: string, key: string): string {
  return CHATROOM_BASE + encodeURIComponent(slug) + (key ? '?key=' + encodeURIComponent(key) : '')
}

const statusStyles: Record<Model['status'], { dot: string; text: string; label: string }> = {
  online: { dot: 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]', text: 'text-emerald-300', label: 'Online, own weights' },
  hosted: { dot: 'bg-sky-400 shadow-[0_0_10px_rgba(56,189,248,0.8)]', text: 'text-sky-300', label: 'Hosted on cloud' },
  offline: { dot: 'bg-rose-400', text: 'text-rose-300', label: 'Offline' },
  private: { dot: 'bg-amber-400', text: 'text-amber-300', label: 'Private machine' },
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

function shortDate(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function maskKey(key: string): string {
  if (key.length <= 14) return key
  return key.slice(0, 9) + '...' + key.slice(-4)
}

function CopyButton({ value, label, primary }: { value: string; label: string; primary?: boolean }) {
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
  const base = primary
    ? 'bg-amber-400 text-[#0e1b3d] hover:bg-amber-300'
    : 'border border-white/12 text-white/60 hover:border-white/25 hover:text-white'
  return (
    <button
      type="button"
      onClick={copy}
      className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[12px] font-medium transition ${base}`}
    >
      {done ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {done ? 'Copied' : label}
    </button>
  )
}

function Sparkline({ days }: { days: Record<string, number> }) {
  const keys = Object.keys(days).sort().slice(-14)
  if (keys.length === 0) return <div className="flex h-10 items-center text-[12px] text-white/25">No requests yet</div>
  const max = Math.max(...keys.map((k) => days[k]), 1)
  return (
    <div className="flex h-10 items-end gap-1">
      {keys.map((k) => (
        <div
          key={k}
          title={k + ': ' + days[k]}
          style={{ height: Math.max(3, (days[k] / max) * 40) }}
          className="w-2.5 rounded-sm bg-amber-400/60 transition hover:bg-amber-300"
        />
      ))}
    </div>
  )
}

function StatCard({ icon: Icon, label, value, hint, tone }: { icon: typeof Zap; label: string; value: string; hint: string; tone: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-5">
      <div className="flex items-center justify-between">
        <p className="text-[12px] font-medium uppercase tracking-wide text-white/35">{label}</p>
        <span className={`grid h-8 w-8 place-items-center rounded-lg ${tone}`}>
          <Icon className="h-4 w-4" />
        </span>
      </div>
      <p className="mt-3 font-mono text-[30px] font-bold tracking-tight">{value}</p>
      <p className="mt-1 text-[12px] text-white/35">{hint}</p>
    </div>
  )
}

export default function UserDashboard() {
  const { username } = useParams<{ username: string }>()
  const { user } = useAuth()
  const location = useLocation()

  const [data, setData] = useState<Payload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<Model | null>(null)

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
  const active = location.hash || '#overview'
  const role = data?.role ?? 'free'
  const roleBadge =
    role === 'owner'
      ? { label: 'Owner', cls: 'border-violet-400/40 bg-violet-400/15 text-violet-200', hint: 'Full access, unlimited models.' }
      : role === 'premium'
        ? { label: 'Premium', cls: 'border-amber-400/40 bg-amber-400/15 text-amber-200', hint: 'Unlimited hosted models.' }
        : { label: 'Free', cls: 'border-white/15 bg-white/5 text-white/60', hint: (data?.count ?? 0) + ' of ' + (data?.limit ?? 2) + ' models used.' }

  const nav = [
    { href: '#overview', label: 'Overview', icon: LayoutDashboard },
    { href: '#models', label: 'Embedded models', icon: Boxes },
    { href: '#guide', label: 'Publishing guide', icon: Zap },
  ]

  return (
    <div className="max-w-6xl mx-auto px-6 py-10 md:py-14">
      <div className="grid gap-10 lg:grid-cols-[220px_1fr]">
        <aside className="hidden lg:block">
          <div className="sticky top-24 space-y-6">
            <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-amber-400 text-[16px] font-bold text-[#0e1b3d]">
                {user.username.slice(0, 1).toUpperCase()}
              </span>
              <div className="min-w-0">
                <p className="truncate text-[14px] font-semibold">{user.username}</p>
                <p className="truncate text-[12px] text-white/40">{user.email || 'InferForge account'}</p>
              </div>
            </div>

            <nav className="space-y-1">
              {nav.map((n) => (
                <a
                  key={n.href}
                  href={n.href}
                  className={`flex items-center gap-3 rounded-xl px-4 py-2.5 text-[14px] font-medium transition ${
                    active === n.href ? 'bg-white/[0.07] text-white' : 'text-white/55 hover:bg-white/[0.05] hover:text-white'
                  }`}
                >
                  <n.icon className="h-4 w-4 text-amber-300/60" />
                  {n.label}
                </a>
              ))}
              <Link
                to="/account"
                className="flex items-center gap-3 rounded-xl px-4 py-2.5 text-[14px] font-medium text-white/55 transition hover:bg-white/[0.05] hover:text-white"
              >
                <Settings className="h-4 w-4 text-amber-300/60" />
                Account settings
              </Link>
            </nav>

            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-4">
              <p className="flex items-center gap-2 text-[12px] font-medium text-white/60">
                <span className={`h-2 w-2 rounded-full ${totals.online > 0 ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]' : 'bg-white/25'}`} />
                {(data?.count ?? 0) > 0 ? (data?.count ?? 0) + ' model' + ((data?.count ?? 0) > 1 ? 's' : '') + ' live' : 'No models live'}
              </p>
              <p className="mt-1.5 text-[11px] leading-relaxed text-white/30">
                {totals.online} on own weights, {(data?.count ?? 0) - totals.online} hosted on cloud.
              </p>
              <div className="mt-3 border-t border-white/[0.07] pt-3">
                <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-bold uppercase tracking-wide ${roleBadge.cls}`}>
                  <Crown className="h-3 w-3" />
                  {roleBadge.label}
                </span>
                <p className="mt-2 text-[11px] text-white/35">{roleBadge.hint}</p>
              </div>
            </div>
          </div>
        </aside>

        <div className="min-w-0">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h1 className="text-[30px] font-bold tracking-tight md:text-[36px]">Dashboard</h1>
              <p className="mt-2 text-[15px] text-white/50">
                Signed in as <span className="font-semibold text-white/80">{user.username}</span>
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-[12px] text-white/35">{data ? 'Checked ' + timeAgo(data.checked_at) : 'Loading'}</span>
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

          <div id="overview" className="mt-8 grid scroll-mt-24 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard icon={Boxes} label="Published" value={loading ? '·' : String(data?.count ?? 0)} hint="via forge embedd" tone="bg-amber-400/15 text-amber-300" />
            <StatCard icon={Radio} label="Own weights live" value={loading ? '·' : String(totals.online)} hint="rest answer from cloud" tone="bg-emerald-400/15 text-emerald-300" />
            <StatCard icon={Activity} label="Requests today" value={loading ? '·' : String(totals.today)} hint="across all models" tone="bg-sky-400/15 text-sky-300" />
            <StatCard icon={Server} label="Requests total" value={loading ? '·' : String(totals.requests)} hint={totals.failed + ' failed'} tone="bg-white/10 text-white/70" />
          </div>

          <div id="models" className="mt-12 flex scroll-mt-24 items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <Boxes className="h-5 w-5 text-amber-300/70" />
              <h2 className="text-[20px] font-semibold tracking-tight">Embedded models</h2>
            </div>
            <span className="text-[12px] text-white/35">{models.length} published</span>
          </div>

          {!loading && models.length === 0 && !error && (
            <div className="mt-6 rounded-2xl border border-dashed border-white/15 bg-white/[0.02] px-6 py-12 text-center">
              <Zap className="mx-auto h-6 w-6 text-amber-300/60" />
              <p className="mt-4 text-[16px] font-semibold">Nothing published yet</p>
              <p className="mx-auto mt-2 max-w-md text-[14px] leading-relaxed text-white/45">
                Connect the CLI with <code className="font-mono text-amber-200">forge connect</code>, then publish a
                model. It shows up here with a live chatroom, status and request counts.
              </p>
              <code className="mt-6 inline-block rounded-xl border border-white/10 bg-black/30 px-4 py-2.5 font-mono text-[13px] text-white/75">
                forge embedd your-model --sdk
              </code>
            </div>
          )}

          <div className="mt-6 space-y-5">
            {models.map((m) => {
              const s = statusStyles[m.status]
              const room = chatroomUrl(m.slug, m.key)
              return (
                <div key={m.slug} className="overflow-hidden rounded-2xl border border-white/10 bg-white/[0.035]">
                  <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.07] px-6 py-5">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2.5">
                        <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${s.dot}`} />
                        <h3 className="truncate text-[18px] font-semibold tracking-tight">{m.model}</h3>
                        <span className={`text-[12px] font-medium ${s.text}`}>{s.label}</span>
                      </div>
                      <a
                        href={room}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-1.5 block truncate font-mono text-[12.5px] text-amber-200/80 transition hover:text-amber-200"
                      >
                        hyperneural.cfd/chatroom/{m.slug}
                      </a>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <a
                        href={room}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 rounded-lg bg-amber-400 px-3.5 py-2 text-[13px] font-semibold text-[#0e1b3d] transition hover:bg-amber-300"
                      >
                        <MessageSquare className="h-3.5 w-3.5" />
                        Open chatroom
                      </a>
                      <a
                        href={m.sdk_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 rounded-lg border border-white/12 px-3.5 py-2 text-[13px] font-medium text-white/70 transition hover:border-white/25 hover:text-white"
                      >
                        SDK
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                      <button
                        type="button"
                        onClick={() => setPendingDelete(m)}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-rose-400/30 px-3.5 py-2 text-[13px] font-medium text-rose-300 transition hover:border-rose-400/60 hover:bg-rose-400/10"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        Delete
                      </button>
                    </div>
                  </div>

                  {m.status === 'hosted' && (
                    <div className="flex items-start gap-3 border-b border-sky-400/15 bg-sky-400/[0.06] px-6 py-3.5 text-[13px] leading-relaxed text-sky-100/80">
                      <Radio className="mt-0.5 h-4 w-4 shrink-0 text-sky-300" />
                      <p>
                        Answering from the cloud around the clock{m.has_persona ? " with this model's persona" : ''}. Your own
                        weights take over whenever <code className="font-mono">forge serve</code> is reachable at a public URL:{' '}
                        <code className="font-mono">forge embedd {m.model} --sdk --endpoint https://your-public-url</code>
                      </p>
                    </div>
                  )}

                  <div className="grid gap-6 px-6 py-5 md:grid-cols-3">
                    <div>
                      <p className="text-[11px] uppercase tracking-wide text-white/30">Requests</p>
                      <div className="mt-2 grid grid-cols-3 gap-3">
                        <div>
                          <p className="font-mono text-[22px] font-bold">{m.requests.total}</p>
                          <p className="text-[11px] text-white/35">total</p>
                        </div>
                        <div>
                          <p className="font-mono text-[22px] font-bold">{m.requests.today}</p>
                          <p className="text-[11px] text-white/35">today</p>
                        </div>
                        <div>
                          <p className="font-mono text-[22px] font-bold text-white/60">{m.requests.failed}</p>
                          <p className="text-[11px] text-white/35">failed</p>
                        </div>
                      </div>
                      <div className="mt-4">
                        <Sparkline days={m.requests.days} />
                      </div>
                      <p className="mt-3 flex items-center gap-1.5 text-[12px] text-white/35">
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
                        <CopyButton value={m.embed} label="Copy embed" primary />
                        <CopyButton value={m.sdk_url} label="Copy URL" />
                      </div>
                    </div>

                    <div className="space-y-3 text-[13px]">
                      <p className="text-[11px] uppercase tracking-wide text-white/30">Details</p>
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-white/40">Embed key</span>
                        <span className="flex items-center gap-2 font-mono text-[12px] text-white/70">
                          <KeyRound className="h-3.5 w-3.5 text-amber-300/60" />
                          {maskKey(m.key)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-white/40">Serving</span>
                        <span className="truncate font-mono text-[12px] text-white/50">{m.status === 'online' ? m.endpoint : (m.hosted_model || 'cloud')}</span>
                      </div>
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-white/40">Published</span>
                        <span className="text-white/70">{shortDate(m.created_at)}</span>
                      </div>
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-white/40">Last status</span>
                        <span className="font-mono text-[12px] text-white/70">{m.requests.last_status ?? 'none'}</span>
                      </div>
                      <div className="pt-1">
                        <CopyButton value={m.key} label="Copy key" />
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          <div id="guide" className="mt-12 scroll-mt-24 rounded-2xl border border-white/10 bg-white/[0.02] p-6">
            <h2 className="text-[16px] font-semibold tracking-tight">Publishing a model</h2>
            <ol className="mt-4 space-y-2.5 text-[14px] leading-relaxed text-white/55">
              <li>
                <span className="font-mono text-amber-200">forge connect</span> links this account to your terminal.
              </li>
              <li>
                <span className="font-mono text-amber-200">forge serve</span> keeps the model loaded and answering.
              </li>
              <li>
                <span className="font-mono text-amber-200">forge embedd &lt;model&gt; --sdk</span> publishes it, opens a
                chatroom on hyperneural.cfd and prints your script tag.
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
      </div>

      {pendingDelete && (
        <DeleteModelDialog
          model={pendingDelete.model}
          slug={pendingDelete.slug}
          session={user.session}
          onClose={() => setPendingDelete(null)}
          onDeleted={() => {
            setPendingDelete(null)
            load(true)
          }}
        />
      )}
    </div>
  )
}
