import { MISSING_JS, SDK_JS } from "../../lib/sdk/template"
import { corsHeaders, javascript, readRecord, slugify } from "../../lib/sdk/store"

const GENERIC = ["inferforge", "inferforge-sdk", "sdk"]

function render(template: string, values: Record<string, string>): string {
  let out = template
  for (const [token, value] of Object.entries(values)) {
    out = out.split("{" + token + "}").join(value)
  }
  return out
}

function safe(value: string): string {
  return String(value || "").replace(/["\\\r\n<>]/g, "")
}

export async function onRequest(context: any): Promise<Response> {
  const request: Request = context.request

  if (request.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders() })
  }
  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response("Method not allowed", { status: 405, headers: corsHeaders() })
  }

  const url = new URL(request.url)
  const segments = (context.params?.path || []) as string[]
  const file = segments.length ? segments[segments.length - 1] : ""

  if (!file.endsWith(".js")) {
    return context.next()
  }

  const name = file.slice(0, -3)
  const slug = slugify(name)
  const origin = url.origin
  const key = safe(url.searchParams.get("key") || "")

  if (GENERIC.indexOf(slug) >= 0) {
    return javascript(
      render(SDK_JS, { ENDPOINT: origin, FALLBACK: "", MODEL: "", API_KEY: key }),
      200,
      { "Cache-Control": "public, max-age=300" }
    )
  }

  const record = await readRecord(context.env.INFERFORGE_SDK, slug)

  if (!record) {
    return javascript(
      render(MISSING_JS, { MODEL: safe(name), SITE: origin }),
      200,
      { "Cache-Control": "no-store" }
    )
  }

  return javascript(
    render(SDK_JS, {
      ENDPOINT: origin,
      FALLBACK: safe(record.fallback || record.endpoint || ""),
      MODEL: safe(record.model || slug),
      API_KEY: key,
    }),
    200,
    { "Cache-Control": "public, max-age=60" }
  )
}
