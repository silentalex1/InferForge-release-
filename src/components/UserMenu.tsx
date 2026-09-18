import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ChevronDown, LayoutDashboard, LogOut, User } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export default function UserMenu() {
  const { user, logout } = useAuth()
  const [open, setOpen] = useState(false)
  const wrap = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  if (!user) return null

  const signOut = () => {
    setOpen(false)
    logout()
    navigate('/')
  }

  return (
    <div ref={wrap} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-[14px] font-semibold transition ${
          open
            ? 'border-amber-400/40 bg-white/[0.07] text-white'
            : 'border-white/15 text-white/85 hover:border-white/30 hover:bg-white/5'
        }`}
      >
        <span className="flex h-5 w-5 items-center justify-center rounded-md bg-amber-400 text-[10px] font-extrabold text-[#0e1b3d]">
          {user.username.slice(0, 1).toUpperCase()}
        </span>
        {user.username}
        <ChevronDown className={`h-3.5 w-3.5 text-white/40 transition ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 z-50 mt-2 w-60 overflow-hidden rounded-xl border border-white/12 bg-[#101e42] shadow-2xl shadow-black/50"
        >
          <div className="border-b border-white/[0.07] px-4 py-3">
            <p className="truncate text-[14px] font-semibold text-white">{user.username}</p>
            <p className="truncate text-[12px] text-white/40">{user.email}</p>
          </div>
          <div className="p-1.5">
            <Link
              to={`/dashboard/${encodeURIComponent(user.username)}`}
              role="menuitem"
              onClick={() => setOpen(false)}
              className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-[14px] font-medium text-white/75 transition hover:bg-white/[0.07] hover:text-white"
            >
              <LayoutDashboard className="h-4 w-4 text-amber-300/70" />
              Dashboard
            </Link>
            <Link
              to="/account"
              role="menuitem"
              onClick={() => setOpen(false)}
              className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-[14px] font-medium text-white/75 transition hover:bg-white/[0.07] hover:text-white"
            >
              <User className="h-4 w-4 text-amber-300/70" />
              Account
            </Link>
            <button
              type="button"
              role="menuitem"
              onClick={signOut}
              className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-[14px] font-medium text-white/75 transition hover:bg-white/[0.07] hover:text-white"
            >
              <LogOut className="h-4 w-4 text-white/35" />
              Log out
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
