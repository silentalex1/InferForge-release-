import { Link } from 'react-router-dom'

export default function Batnight() {
  return (
    <div className="min-h-[calc(100vh-64px)] bg-[#0A0A0B] flex items-center justify-center px-6 py-16">
      <div className="max-w-2xl w-full text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-white/[0.04] text-emerald-400 text-xs font-semibold mb-6">batnight.inferforge.org • Live</div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-4">Batnight</h1>
        <p className="text-white/50 leading-6">Project for <span className="text-white font-mono">/batnight</span> — hosted on <span className="text-white">batnight.inferforge.org</span></p>
        <div className="mt-8 flex justify-center gap-3">
          <Link to="/" className="px-5 py-2.5 rounded-xl bg-white text-black text-sm font-semibold">Go home</Link>
          <a href="https://inferforge.org" className="px-5 py-2.5 rounded-xl border border-white/10 text-white/70 text-sm">inferforge.org</a>
        </div>
      </div>
    </div>
  )
}
