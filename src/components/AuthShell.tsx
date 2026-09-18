import { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Check } from 'lucide-react'

type Props = {
  title: string
  subtitle: string
  children: ReactNode
  footer: ReactNode
}

const perks = [
  'Publish models to inferforge.org/sdk/<model>.js',
  'Manage per-model embed keys from one place',
  'Link the CLI to your account with forge connect',
  'Keep every weight and every prompt on your own hardware',
]

export default function AuthShell({ title, subtitle, children, footer }: Props) {
  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[520px] bg-[radial-gradient(ellipse_55%_50%_at_50%_0%,rgba(251,191,36,0.12),transparent_70%)]" />

      <div className="relative max-w-6xl mx-auto grid gap-14 px-6 py-16 md:py-24 lg:grid-cols-2 lg:gap-20">
        <div className="hidden lg:block">
          <span className="inline-flex items-center rounded-full border border-amber-400/25 bg-amber-400/10 px-3 py-1 text-[12px] font-semibold text-amber-200">
            InferForge account
          </span>
          <h2 className="mt-6 text-[34px] font-bold leading-tight tracking-tight">
            One account for every model you ship.
          </h2>
          <p className="mt-4 text-[16px] leading-relaxed text-white/55">
            The CLI runs fine on its own. An account is what lets you publish a model to the web and
            keep track of what is live.
          </p>
          <ul className="mt-8 space-y-3.5">
            {perks.map((p) => (
              <li key={p} className="flex gap-3 text-[14px] leading-relaxed text-white/65">
                <Check className="mt-[3px] h-4 w-4 shrink-0 text-amber-300/70" />
                <span>{p}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="w-full max-w-md justify-self-center lg:justify-self-end">
          <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-7 md:p-8">
            <h1 className="text-[24px] font-bold tracking-tight">{title}</h1>
            <p className="mt-2 text-[14px] leading-relaxed text-white/50">{subtitle}</p>
            <div className="mt-7">{children}</div>
          </div>
          <div className="mt-5 text-center text-[14px] text-white/45">{footer}</div>
          <div className="mt-3 text-center text-[13px] text-white/30">
            <Link to="/" className="transition hover:text-white/60">
              Back to inferforge.org
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
