import { Link } from 'react-router-dom'
import { Check } from 'lucide-react'

const freeItems = [
  'Host up to 2 AI models',
  'Run, train and merge models locally',
  'Hosted SDK and chatroom per model',
  'Import from Ollama and HuggingFace',
  'Community support',
]

const premiumItems = [
  'Host unlimited AI models',
  'Everything in Free',
  'Advanced merging (Procrustes, Fisher, SVD)',
  'Full training system',
  'Priority support',
  'Performance analytics',
]

const sourceItems = [
  'Complete source code',
  'Full architecture',
  'Commercial rights',
  'Resell rights',
  'Custom modifications',
  'Dedicated support',
]

function TickList({ items }: { items: string[] }) {
  return (
    <ul className="space-y-3">
      {items.map((item, i) => (
        <li
          key={item}
          className={`flex items-start gap-2.5 text-[13.5px] leading-5 ${i === 0 ? 'font-semibold text-white' : 'text-white/65'}`}
        >
          <span className="mt-0.5 flex h-4 w-4 flex-shrink-0 items-center justify-center rounded-full bg-amber-400/15">
            <Check className="h-3 w-3 text-amber-300" strokeWidth={3} />
          </span>
          <span>{item}</span>
        </li>
      ))}
    </ul>
  )
}

export default function Pricing() {
  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[520px] bg-[radial-gradient(ellipse_55%_50%_at_50%_0%,rgba(251,191,36,0.1),transparent_70%)]" />

      <div className="relative max-w-6xl mx-auto px-6 py-16 md:py-20">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <p className="mb-3 text-[12px] font-semibold uppercase tracking-widest text-white/30">Pricing</p>
          <h1 className="mb-4 text-[38px] font-extrabold leading-none tracking-tight md:text-[48px]">
            Simple, honest pricing.
          </h1>
          <p className="text-[15px] leading-6 text-white/50">
            Start free with two hosted models. Upgrade when you need more. Own the source when you are ready.
          </p>
        </div>

        <div className="mx-auto grid max-w-5xl items-stretch gap-5 md:grid-cols-3">
          <div className="flex flex-col rounded-2xl border border-white/10 bg-white/[0.035] p-7">
            <p className="mb-1 text-[14px] font-semibold text-white">Free</p>
            <p className="mb-6 text-[36px] font-extrabold tracking-tight text-white">Free</p>
            <div className="flex-1">
              <TickList items={freeItems} />
            </div>
            <Link
              to="/download"
              className="mt-8 inline-flex w-full items-center justify-center rounded-xl border border-white/15 px-5 py-3 text-[14px] font-semibold text-white/85 transition hover:border-white/30 hover:bg-white/5"
            >
              Install Now
            </Link>
          </div>

          <div className="relative flex flex-col rounded-2xl border border-amber-400/40 bg-gradient-to-b from-amber-400/[0.1] to-white/[0.03] p-7 shadow-[0_16px_48px_rgba(251,191,36,0.14)]">
            <span className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full bg-amber-400 px-3 py-1 text-[10px] font-extrabold uppercase tracking-widest text-[#0e1b3d] shadow-md">
              Most Popular
            </span>
            <p className="mb-1 text-[14px] font-semibold text-white">Premium</p>
            <div className="mb-6 flex items-baseline gap-2">
              <span className="text-[36px] font-extrabold tracking-tight text-white">$10</span>
              <span className="text-[14px] text-white/40">Lifetime</span>
            </div>
            <div className="flex-1">
              <TickList items={premiumItems} />
            </div>
            <a
              href="mailto:hello@inferforge.dev"
              className="mt-8 inline-flex w-full items-center justify-center rounded-xl bg-amber-400 px-5 py-3 text-[14px] font-bold text-[#0e1b3d] transition hover:bg-amber-300"
            >
              Upgrade to Premium
            </a>
          </div>

          <div className="flex flex-col rounded-2xl border border-white/10 bg-white/[0.035] p-7">
            <p className="mb-1 text-[14px] font-semibold text-white">Source Code</p>
            <div className="mb-6 flex items-baseline gap-2">
              <span className="text-[36px] font-extrabold tracking-tight text-white">$2k-6k</span>
              <span className="text-[14px] text-white/40">One-time</span>
            </div>
            <div className="flex-1">
              <TickList items={sourceItems} />
            </div>
            <a
              href="mailto:hello@inferforge.dev"
              className="mt-8 inline-flex w-full items-center justify-center rounded-xl border border-white/15 px-5 py-3 text-[14px] font-semibold text-white/85 transition hover:border-white/30 hover:bg-white/5"
            >
              Contact Site Owner
            </a>
            <div className="mt-4 text-center text-[12px] leading-5">
              <p className="font-medium text-white/60">We&apos;ll talk.</p>
              <p className="text-white/35">
                add{' '}
                <button
                  type="button"
                  onClick={async () => {
                    await navigator.clipboard.writeText('jahmiseryx')
                    const el = document.getElementById('copy-jahmiseryx')
                    if (el) {
                      el.textContent = 'copied!'
                      setTimeout(() => (el.textContent = 'jahmiseryx'), 1200)
                    }
                  }}
                  className="font-semibold text-amber-300 underline decoration-amber-300/40 underline-offset-4 transition hover:text-amber-200"
                >
                  <span id="copy-jahmiseryx">jahmiseryx</span>
                </button>{' '}
                on discord.
              </p>
            </div>
          </div>
        </div>

        <p className="mt-10 text-center text-[12px] text-white/25">
          Questions?{' '}
          <a href="mailto:hello@inferforge.dev" className="text-white/50 underline underline-offset-4 transition hover:text-white">
            hello@inferforge.dev
          </a>{' '}
          ·{' '}
          <Link to="/docs" className="text-white/50 underline underline-offset-4 transition hover:text-white">
            Read the docs
          </Link>
        </p>
      </div>
    </div>
  )
}
