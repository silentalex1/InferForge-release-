import { useEffect, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { Menu, X } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import UserMenu from './UserMenu'

const links = [
  { to: '/docs', label: 'Docs' },
  { to: '/models', label: 'Models' },
  { to: '/download', label: 'Download' },
  { to: '/pricing', label: 'Pricing' },
]

export default function Navbar() {
  const [open, setOpen] = useState(false)
  const { user, logout } = useAuth()
  const location = useLocation()

  useEffect(() => {
    setOpen(false)
  }, [location.pathname])

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `text-[14px] font-medium transition ${isActive ? 'text-white' : 'text-white/55 hover:text-white/90'}`

  return (
    <nav className="sticky top-0 z-50 border-b border-white/[0.07] bg-[#0e1b3d]/85 backdrop-blur-xl">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex h-[64px] items-center justify-between gap-6">
          <Link to="/" className="flex shrink-0 items-center gap-3">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-white">
              <span className="text-[11px] font-extrabold tracking-tighter text-black">IF</span>
            </span>
            <span className="text-[15px] font-bold tracking-tight text-white">InferForge</span>
            <span className="hidden items-center rounded-full border border-amber-400/20 bg-amber-400/15 px-2 py-0.5 text-[10px] font-semibold tracking-wide text-amber-300 sm:inline-flex">
              v0.2.4
            </span>
          </Link>

          <div className="hidden items-center gap-7 md:flex">
            {links.map((l) => (
              <NavLink key={l.to} to={l.to} className={linkClass}>
                {l.label}
              </NavLink>
            ))}
          </div>

          <div className="hidden items-center gap-3 md:flex">
            {user ? (
              <UserMenu />
            ) : (
              <>
                <Link
                  to="/login"
                  className="rounded-lg px-3 py-2 text-[14px] font-medium text-white/65 transition hover:text-white"
                >
                  Login
                </Link>
                <Link
                  to="/create-account"
                  className="rounded-lg bg-amber-400 px-4 py-2 text-[14px] font-semibold text-[#0e1b3d] transition hover:bg-amber-300"
                >
                  Create account
                </Link>
              </>
            )}
          </div>

          <button
            type="button"
            aria-label={open ? 'Close menu' : 'Open menu'}
            onClick={() => setOpen((v) => !v)}
            className="rounded-lg p-2 text-white/70 transition hover:bg-white/5 hover:text-white md:hidden"
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {open && (
        <div className="border-t border-white/[0.07] bg-[#0e1b3d] md:hidden">
          <div className="max-w-6xl mx-auto space-y-1 px-6 py-4">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                className="block rounded-lg px-3 py-2.5 text-[15px] font-medium text-white/70 transition hover:bg-white/5 hover:text-white"
              >
                {l.label}
              </NavLink>
            ))}
            <div className="mt-3 flex flex-col gap-2 border-t border-white/[0.07] pt-4">
              {user ? (
                <>
                  <p className="px-1 pb-1 text-[12px] text-white/35">Signed in as {user.username}</p>
                  <Link
                    to={`/dashboard/${encodeURIComponent(user.username)}`}
                    className="rounded-lg bg-amber-400 px-4 py-2.5 text-center text-[15px] font-semibold text-[#0e1b3d]"
                  >
                    Dashboard
                  </Link>
                  <Link
                    to="/account"
                    className="rounded-lg border border-white/15 px-4 py-2.5 text-center text-[15px] font-semibold text-white/85"
                  >
                    Account
                  </Link>
                  <button
                    type="button"
                    onClick={logout}
                    className="rounded-lg px-4 py-2.5 text-center text-[15px] font-medium text-white/55"
                  >
                    Log out
                  </button>
                </>
              ) : (
                <>
                  <Link
                    to="/login"
                    className="rounded-lg border border-white/15 px-4 py-2.5 text-center text-[15px] font-semibold text-white/85"
                  >
                    Login
                  </Link>
                  <Link
                    to="/create-account"
                    className="rounded-lg bg-amber-400 px-4 py-2.5 text-center text-[15px] font-semibold text-[#0e1b3d]"
                  >
                    Create account
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </nav>
  )
}
