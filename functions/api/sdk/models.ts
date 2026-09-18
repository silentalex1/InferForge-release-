import {
  corsHeaders,
  json,
  listRecords,
  probeUpstream,
  readStats,
  today,
} from "../../../lib/sdk/store"
import type { SdkRecord } from "../../../lib/sdk/store"

function isPrivate(endpoint: string): boolean {
  let host = ""
  try {
    host = new URL(endpoint).hostname.toLowerCase().replace(/^\[|\]$/g, "")
  } catch {
    return true
  }
  if (host === "localhost" || host === "0.0.0.0" || host === "::1" || host.endsWith(".local")) return true
  return /^127\.|^10\.|^192\.168\.|^169\.254\.|^172\.(1[6-9]|2[0-9]|3[01])\./.test(host)
}

export async function onRequest(context: any): Promise<Response> {
  const request: Request = context.request

  if (request.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders() })
  }
  if (request.method !== "GET") {
    return json({ error: "method-not-allowed" }, 405)
  }

  const kv = context.env.INFERFORGE_SDK
  if (!kv) {
    return json({ error: "registry-unavailable" }, 503)
  }

  const url = new URL(request.url)
  const owner = (url.searchParams.get("owner") || "").trim().toLowerCase()
  const probe = url.searchParams.get("probe") !== "0"

  const all = await listRecords(kv)
  const mine = owner
    ? all.filter((r: SdkRecord) => (r.owner || "").toLowerCase() === owner)
    : all

  const day = today()

  const models = await Promise.all(
    mine.map(async (record) => {
      const stats = await readStats(kv, record.slug)
      const endpoint = record.endpoint || ""
      const unreachable = isPrivate(endpoint)
      const online = unreachable ? false : probe ? await probeUpstream(endpoint) : false

      return {
        model: record.model,
        slug: record.slug,
        owner: record.owner || null,
        endpoint,
        sdk_url: url.origin + "/sdk/" + record.slug + ".js",
        embed:
          '<script src="' +
          url.origin +
          "/sdk/" +
          record.slug +
          ".js?key=" +
          record.key +
          '&mount=%23inferforge-chat"></script>',
        key: record.key,
        created_at: record.created_at,
        updated_at: record.updated_at,
        status: online ? "online" : unreachable ? "private" : "offline",
        online,
        requests: {
          total: stats.total,
          ok: stats.ok,
          failed: stats.failed,
          today: stats.days[day] || 0,
          days: stats.days,
          last_request_at: stats.last_request_at,
          last_status: stats.last_status,
        },
      }
    })
  )

  models.sort((a, b) => (a.created_at < b.created_at ? 1 : -1))

  const totals = models.reduce(
    (acc, m) => {
      acc.requests += m.requests.total
      acc.today += m.requests.today
      acc.failed += m.requests.failed
      if (m.online) acc.online += 1
      return acc
    },
    { requests: 0, today: 0, failed: 0, online: 0 }
  )

  return json(
    {
      owner: owner || null,
      count: models.length,
      totals,
      checked_at: new Date().toISOString(),
      models,
    },
    200,
    { "Cache-Control": "no-store" }
  )
}
