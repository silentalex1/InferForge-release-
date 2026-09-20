import { Link } from 'react-router-dom'

const product = [
  { to: '/download', label: 'Download' },
  { to: '/docs', label: 'Documentation' },
  { to: '/models', label: 'Models' },
  { to: '/pricing', label: 'Pricing' },
]

const account = [
  { to: '/login', label: 'Login' },
  { to: '/create-account', label: 'Create account' },
]

const external = [
  { href: 'https://github.com/silentalex1/HyperNeural', label: 'GitHub' },
  { href: 'https://discord.gg/Nc9fqvRM68', label: 'Discord' },
  { href: 'mailto:hello@inferforge.dev', label: 'Support' },
]

export default function Footer() {
  const year = new Date().getFullYear()
  return (
    <footer className="border-t border-white/[0.07] bg-[#0b1530]">
      <div className="max-w-6xl mx-auto px-6 py-14">
        <div className="grid gap-10 md:grid-cols-[1.4fr_1fr_1fr_1fr]">
          <div>
            <Link to="/" className="flex items-center gap-2.5">
              <span className="flex h-6 w-6 items-center justify-center rounded-md bg-white">
                <span className="text-[10px] font-extrabold tracking-tighter text-black">IF</span>
              </span>
              <span className="text-sm font-bold tracking-tight text-white">InferForge</span>
            </Link>
            <p className="mt-4 max-w-xs text-[13px] leading-relaxed text-white/40">
              A self-hosted local AI platform for running, training, merging and deploying language
              models on hardware you own.
            </p>
          </div>

          <div>
            <h3 className="text-[13px] font-semibold tracking-tight text-white/80">Product</h3>
            <ul className="mt-4 space-y-2.5">
              {product.map((l) => (
                <li key={l.to}>
                  <Link to={l.to} className="text-[13px] text-white/45 transition hover:text-white/80">
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="text-[13px] font-semibold tracking-tight text-white/80">Account</h3>
            <ul className="mt-4 space-y-2.5">
              {account.map((l) => (
                <li key={l.to}>
                  <Link to={l.to} className="text-[13px] text-white/45 transition hover:text-white/80">
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="text-[13px] font-semibold tracking-tight text-white/80">Community</h3>
            <ul className="mt-4 space-y-2.5">
              {external.map((l) => (
                <li key={l.href}>
                  <a
                    href={l.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[13px] text-white/45 transition hover:text-white/80"
                  >
                    {l.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-3 border-t border-white/[0.07] pt-6 sm:flex-row">
          <span className="text-[12px] text-white/30">© {year} InferForge</span>
          <span className="font-mono text-[12px] text-white/25">v0.2.4</span>
        </div>
      </div>
    </footer>
  )
}
