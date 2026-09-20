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
  system?: string
  hosted_model?: string
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

export const OWNER_PREFIX = "owner:"

export function ownerKey(owner: string): string {
  return OWNER_PREFIX + String(owner || "").trim().toLowerCase()
}

export async function ownerSlugs(kv: any, owner: string): Promise<string[]> {
  if (!kv || !owner) return []
  const key = ownerKey(owner)
  const raw = await kv.get(key)
  if (raw) {
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) return parsed.map(String)
    } catch {}
  }
  const all = await listRecords(kv)
  const slugs = all
    .filter((r) => (r.owner || "").toLowerCase() === owner.trim().toLowerCase())
    .map((r) => r.slug)
  await kv.put(key, JSON.stringify(slugs))
  return slugs
}

export async function addOwnerSlug(kv: any, owner: string, slug: string): Promise<void> {
  if (!kv || !owner) return
  const slugs = await ownerSlugs(kv, owner)
  if (slugs.includes(slug)) return
  slugs.push(slug)
  await kv.put(ownerKey(owner), JSON.stringify(slugs))
}

export async function removeOwnerSlug(kv: any, owner: string, slug: string): Promise<void> {
  if (!kv || !owner) return
  const slugs = await ownerSlugs(kv, owner)
  const next = slugs.filter((s) => s !== slug)
  if (next.length === slugs.length) return
  await kv.put(ownerKey(owner), JSON.stringify(next))
}

export type AccountRole = "owner" | "premium" | "free"

export async function accountRole(kv: any, owner: string): Promise<AccountRole> {
  if (!kv || !owner) return "free"
  const name = owner.trim().toLowerCase()
  const role = await kv.get("role:" + name)
  if (role === "owner" || role === "premium") return role
  const legacy = await kv.get("premium:" + name)
  return legacy ? "premium" : "free"
}

export async function isPremium(kv: any, owner: string): Promise<boolean> {
  return (await accountRole(kv, owner)) !== "free"
}

export const SESSION_TTL = 2592000

export async function createSession(kv: any, username: string): Promise<string> {
  if (!kv || !username) return ""
  const token = "sess-" + crypto.randomUUID().replace(/-/g, "")
  await kv.put("session:" + token, username.trim().toLowerCase(), { expirationTtl: SESSION_TTL })
  return token
}

export async function sessionUser(kv: any, token: string): Promise<string> {
  if (!kv || !token) return ""
  const raw = await kv.get("session:" + String(token).trim())
  return raw || ""
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

export const HOSTED_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

export const HOSTED_MODELS = [
  HOSTED_MODEL,
  "@cf/meta/llama-3.1-8b-instruct",
  "@cf/meta/llama-3.1-70b-instruct",
  "@cf/qwen/qwen2.5-coder-32b-instruct",
  "@cf/mistral/mistral-7b-instruct-v0.2",
]

export function isPrivateHost(endpoint: string): boolean {
  let host = ""
  try {
    host = new URL(endpoint).hostname.toLowerCase().replace(/^\[|\]$/g, "")
  } catch {
    return true
  }
  if (host === "localhost" || host === "0.0.0.0" || host === "::1" || host.endsWith(".local")) return true
  return /^127\.|^10\.|^192\.168\.|^169\.254\.|^172\.(1[6-9]|2[0-9]|3[01])\./.test(host)
}

export function withPersona(messages: any[], system: string): any[] {
  const list = Array.isArray(messages) ? messages : []
  if (!system) return list
  if (list.some((m) => m && m.role === "system")) return list
  return [{ role: "system", content: system }, ...list]
}

export function openAiChunk(id: string, model: string, created: number, delta: Record<string, string>, finish: string | null) {
  return {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  }
}

export async function hostedChat(ai: any, record: SdkRecord, body: any, extra: Record<string, string> = {}): Promise<Response> {
  const model = HOSTED_MODELS.includes(record.hosted_model || "") ? (record.hosted_model as string) : HOSTED_MODEL
  const messages = withPersona(body?.messages, record.system || "")
  const opts: Record<string, unknown> = { messages, stream: !!body?.stream }
  if (typeof body?.max_tokens === "number") opts.max_tokens = body.max_tokens
  if (typeof body?.temperature === "number") opts.temperature = body.temperature
  const id = "chatcmpl-" + crypto.randomUUID().replace(/-/g, "").slice(0, 12)
  const created = Math.floor(Date.now() / 1000)

  if (!body?.stream) {
    const r: any = await ai.run(model, opts)
    const content = typeof r === "string" ? r : String(r?.response ?? "")
    return json(
      {
        id,
        object: "chat.completion",
        created,
        model: record.model,
        choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
        usage: r?.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      },
      200,
      extra
    )
  }

  const source: ReadableStream = await ai.run(model, opts)
  const enc = new TextEncoder()
  const dec = new TextDecoder()
  let buf = ""
  const sse = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, ctrl) {
      buf += dec.decode(chunk, { stream: true })
      const lines = buf.split("\n")
      buf = lines.pop() || ""
      for (const line of lines) {
        const t = line.trim()
        if (!t.startsWith("data:")) continue
        const payload = t.slice(5).trim()
        if (payload === "[DONE]") continue
        try {
          const j = JSON.parse(payload)
          const tok = typeof j === "string" ? j : String(j?.response ?? "")
          if (tok) ctrl.enqueue(enc.encode("data: " + JSON.stringify(openAiChunk(id, record.model, created, { content: tok }, null)) + "\n\n"))
        } catch {}
      }
    },
    flush(ctrl) {
      ctrl.enqueue(enc.encode("data: " + JSON.stringify(openAiChunk(id, record.model, created, {}, "stop")) + "\n\n"))
      ctrl.enqueue(enc.encode("data: [DONE]\n\n"))
    },
  })
  return new Response(source.pipeThrough(sse), {
    status: 200,
    headers: corsHeaders({ "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-store", ...extra }),
  })
}
