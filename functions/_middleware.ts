const PASSTHROUGH = ["/api/", "/assets/", "/sdk/"]
const EXACT = ["/v1/chat/completions", "/v1/models"]

export async function onRequest(context: any) {
  const url = new URL(context.request.url)
  const path = url.pathname

  if (EXACT.indexOf(path) >= 0) return context.next()
  if (PASSTHROUGH.some((prefix) => path.startsWith(prefix))) return context.next()
  if (path.includes(".")) return context.next()

  const res = await context.next()
  if (res.status !== 404) return res

  const asset = await context.env.ASSETS.fetch(new Request(new URL("/index.html", url).toString()))
  return new Response(asset.body, { status: 200, headers: asset.headers })
}
