
export async function onRequest(context: any): Promise<Response> {
  const url = new URL(context.request.url)
  return Response.redirect(url.origin + "/InferForgeInstaller.exe", 302)
}
