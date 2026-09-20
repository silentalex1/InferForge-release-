import {
  corsHeaders,
  deleteRecord,
  deleteStats,
  json,
  readRecord,
  removeOwnerSlug,
  sessionUser,
  slugify,
} from "../../../lib/sdk/store"

export async function onRequest(context: any): Promise<Response> {
  const request: Request = context.request
  const kv = context.env.INFERFORGE_SDK

  if (request.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders({ "Access-Control-Allow-Headers": "Content-Type, X-Forge-Session" }) })
  }
  if (request.method !== "POST") {
    return json({ error: "method-not-allowed" }, 405)
  }
  if (!kv) {
    return json({ error: "registry-unavailable", message: "The model registry is not configured." }, 503)
  }

  let body: any
  try {
    body = await request.json()
  } catch {
    return json({ error: "invalid-json" }, 400)
  }

  const token = request.headers.get("x-forge-session") || ""
  const username = await sessionUser(kv, token)
  if (!username) {
    return json({ error: "not-signed-in", message: "Log in again to delete a model." }, 401)
  }

  const slug = slugify(body?.slug || body?.model || "")
  const confirm = slugify(body?.confirm || "")
  const record = await readRecord(kv, slug)
  if (!record) {
    return json({ error: "not-found", message: "No published model named '" + slug + "'." }, 404)
  }

  if ((record.owner || "").toLowerCase() !== username) {
    return json({ error: "forbidden", message: "This model belongs to another account." }, 403)
  }

  if (confirm !== slug) {
    return json(
      { error: "name-mismatch", message: "Type the model name exactly to confirm deletion." },
      400
    )
  }

  await deleteRecord(kv, slug)
  await deleteStats(kv, slug)
  await removeOwnerSlug(kv, record.owner, slug)

  return json({ ok: true, slug, model: record.model })
}
