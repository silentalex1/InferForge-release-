export interface SdkRecord {
  model: string
  slug: string
  key: string
  endpoint: string
  fallback: string
  owner: string
  token_hash: string
  created_at: string
  updated_at: string
}

export const KEY_PREFIX = "sdk:"

export function slugify(model: string): string {
  const slug = String(model || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "")
  return slug || "model"
}

export function cleanEndpoint(url: string): string {
  return String(url || "").trim().replace(/\/+$/, "")
}

export function isHttpUrl(url: string): boolean {
  try {
    const parsed = new URL(url)
    return parsed.protocol === "http:" || parsed.protocol === "https:"
  } catch {
    return false
  }
}

export function corsHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Forge-Token, x-api-key",
    "Access-Control-Max-Age": "86400",
    ...extra,
  }
}

export function json(body: unknown, status = 200, extra: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: corsHeaders({ "Content-Type": "application/json; charset=utf-8", ...extra }),
  })
}

export function javascript(body: string, status = 200, extra: Record<string, string> = {}): Response {
  return new Response(body, {
    status,
    headers: corsHeaders({
      "Content-Type": "application/javascript; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
      ...extra,
    }),
  })
}

export async function hashToken(token: string): Promise<string> {
  const data = new TextEncoder().encode(String(token || ""))
  const digest = await crypto.subtle.digest("SHA-256", data)
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
}

export function timingSafeEqual(a: string, b: string): boolean {
  const left = String(a || "")
  const right = String(b || "")
  if (left.length !== right.length) return false
  let diff = 0
  for (let i = 0; i < left.length; i++) diff |= left.charCodeAt(i) ^ right.charCodeAt(i)
  return diff === 0
}

export function bearer(request: Request): string {
  const auth = request.headers.get("authorization") || ""
  if (auth.toLowerCase().startsWith("bearer ")) return auth.slice(7).trim()
  return request.headers.get("x-api-key") || ""
}

export async function readRecord(kv: any, slug: string): Promise<SdkRecord | null> {
  if (!kv) return null
  const raw = await kv.get(KEY_PREFIX + slug)
  if (!raw) return null
  try {
    return JSON.parse(raw) as SdkRecord
  } catch {
    return null
  }
}

export async function writeRecord(kv: any, record: SdkRecord): Promise<void> {
  await kv.put(KEY_PREFIX + record.slug, JSON.stringify(record))
}

export async function deleteRecord(kv: any, slug: string): Promise<void> {
  await kv.delete(KEY_PREFIX + slug)
}

export async function listRecords(kv: any, limit = 200): Promise<SdkRecord[]> {
  if (!kv) return []
  const listed = await kv.list({ prefix: KEY_PREFIX, limit })
  const out: SdkRecord[] = []
  for (const entry of listed.keys || []) {
    const record = await readRecord(kv, String(entry.name).slice(KEY_PREFIX.length))
    if (record) out.push(record)
  }
  return out
}

export interface SdkStats {
  slug: string
  total: number
  ok: number
  failed: number
  days: Record<string, number>
  last_request_at: string | null
  last_status: number | null
}

export const STATS_PREFIX = "stats:"

export const DAY_WINDOW = 14

export function emptyStats(slug: string): SdkStats {
  return { slug, total: 0, ok: 0, failed: 0, days: {}, last_request_at: null, last_status: null }
}

export function today(): string {
  return new Date().toISOString().slice(0, 10)
}

export async function readStats(kv: any, slug: string): Promise<SdkStats> {
  if (!kv) return emptyStats(slug)
  const raw = await kv.get(STATS_PREFIX + slug)
  if (!raw) return emptyStats(slug)
  try {
    return { ...emptyStats(slug), ...(JSON.parse(raw) as SdkStats) }
  } catch {
    return emptyStats(slug)
  }
}

export async function bumpStats(kv: any, slug: string, status: number): Promise<void> {
  if (!kv) return
  const stats = await readStats(kv, slug)
  const day = today()
  const ok = status >= 200 && status < 400

  stats.total += 1
  if (ok) stats.ok += 1
  else stats.failed += 1
  stats.days[day] = (stats.days[day] || 0) + 1
  stats.last_request_at = new Date().toISOString()
  stats.last_status = status

  const keys = Object.keys(stats.days).sort()
  while (keys.length > DAY_WINDOW) {
    const oldest = keys.shift()
    if (oldest) delete stats.days[oldest]
  }

  await kv.put(STATS_PREFIX + slug, JSON.stringify(stats))
}

export async function deleteStats(kv: any, slug: string): Promise<void> {
  if (!kv) return
  await kv.delete(STATS_PREFIX + slug)
}

export async function probeUpstream(endpoint: string, timeoutMs = 4000): Promise<boolean> {
  const url = cleanEndpoint(endpoint)
  if (!isHttpUrl(url)) return false
  try {
    const res = await fetch(url + "/health", { signal: AbortSignal.timeout(timeoutMs) })
    return res.ok
  } catch {
    return false
  }
}

export function publicView(record: SdkRecord) {
  return {
    model: record.model,
    slug: record.slug,
    owner: record.owner || null,
    sdk_url: "/sdk/" + record.slug + ".js",
    created_at: record.created_at,
    updated_at: record.updated_at,
  }
}
