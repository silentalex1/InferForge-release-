export default {
  async fetch(request: Request, env: { ASSETS: { fetch: (req: Request) => Promise<Response> } }): Promise<Response> {
    const url = new URL(request.url)
    let path = url.pathname

    if (url.hostname.startsWith("api.") || path.startsWith("/api/")) {
      if (path === "/" || path === "") path = "/index.html"
      if (path.includes("Inferforge") || path.includes("inferforge")) {
        return new Response("{inferforge.model;active}", {
          headers: { "Content-Type": "text/plain; charset=utf-8", "Access-Control-Allow-Origin": "*" },
        })
      }
    }

    if (path === "/" || path === "") path = "/index.html"

    if (path.startsWith("/chatroom/")) {
      const room = await env.ASSETS.fetch(new URL("/chatroom/index.html", url.origin))
      if (room.status !== 404) {
        const headers = new Headers(room.headers)
        headers.set("Access-Control-Allow-Origin", "*")
        return new Response(room.body, { status: 200, headers })
      }
    }

    if (path !== "/" && path.split("/").filter(Boolean).length === 1 && !/^\/(assets|pypi|install|admin-panel|api)\b/.test(path) && !/\.(js|css|html|json|svg|png|jpg|ico|woff2?|ttf|map)$/i.test(path)) {
      const sdk = await env.ASSETS.fetch(new URL("/sdk.html", url.origin))
      if (sdk.status !== 404) {
        const headers = new Headers(sdk.headers)
        headers.set("Access-Control-Allow-Origin", "*")
        return new Response(sdk.body, { status: 200, headers })
      }
    }

    const asset = await env.ASSETS.fetch(new URL(path, url.origin))
    if (asset.status !== 404) {
      const headers = new Headers(asset.headers)
      headers.set("Access-Control-Allow-Origin", "*")
      return new Response(asset.body, { status: asset.status, headers })
    }

    return new Response("Not Found", { status: 404 })
  },
}
