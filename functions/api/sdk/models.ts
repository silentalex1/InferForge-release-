import {
  corsHeaders,
  HOSTED_MODEL,
  isPrivateHost,
  json,
  listRecords,
  probeUpstream,
  readStats,
  slugify,
  today,
} from "../../../lib/sdk/store"
import type { SdkRecord } from "../../../lib/sdk/store"

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
  const slugFilter = (url.searchParams.get("slug") || "").trim()
  const wanted = slugFilter ? slugify(slugFilter) : ""
  const probe = url.searchParams.get("probe") !== "0"

  const all = await listRecords(kv)
  const mine = all.filter((r: SdkRecord) => {
    if (owner && (r.owner || "").toLowerCase() !== owner) return false
    if (wanted && r.slug !== wanted) return false
    return true
  })

  const day = today()

  const models = await Promise.all(
    mine.map(async (record) => {
      const stats = await readStats(kv, record.slug)
      const endpoint = record.endpoint || ""
      const unreachable = isPrivateHost(endpoint)
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
        status: online ? "online" : "hosted",
        online,
        upstream_private: unreachable,
        hosted_model: record.hosted_model || HOSTED_MODEL,
        has_persona: Boolean(record.system),
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
