import { corsHeaders, json, listRecords } from "../../lib/sdk/store"

export async function onRequest(context: any): Promise<Response> {
  const request: Request = context.request

  if (request.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders() })
  }
  if (request.method !== "GET") {
    return json({ error: "method-not-allowed" }, 405)
  }

  const records = await listRecords(context.env.INFERFORGE_SDK)
  return json({
    object: "list",
    data: records.map((record) => ({
      id: record.model,
      object: "model",
      created: Math.floor(new Date(record.created_at || Date.now()).getTime() / 1000),
      owned_by: record.owner || "inferforge",
      sdk_url: "/sdk/" + record.slug + ".js",
    })),
  })
}
