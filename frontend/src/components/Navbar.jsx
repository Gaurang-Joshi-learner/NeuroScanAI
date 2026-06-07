import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { Brain, LayoutDashboard, Upload, LogOut, ChevronDown } from 'lucide-react'
import { useState } from 'react'
import clsx from 'clsx'

export default function Navbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)
  const isActive = p => pathname === p

  return (
    <nav className="bg-gray-950/80 backdrop-blur-md border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/dashboard" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-violet-600 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-white text-lg">NeuroScanAI</span>
          </Link>

          <div className="flex items-center gap-1">
            {[{ to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
              { to: '/analyze',   icon: Upload,          label: 'Analyze'   }].map(({ to, icon: Icon, label }) => (
              <Link key={to} to={to} className={clsx(
                'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                isActive(to) ? 'bg-violet-900/50 text-violet-300' : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              )}>
                <Icon className="w-4 h-4" />{label}
              </Link>
            ))}
          </div>

          <div className="relative">
            <button onClick={() => setOpen(!open)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-800 transition-colors">
              <div className="w-8 h-8 bg-violet-800 rounded-full flex items-center justify-center">
                <span className="text-violet-200 text-sm font-bold">
                  {user?.full_name?.[0]?.toUpperCase() || user?.email?.[0]?.toUpperCase()}
                </span>
              </div>
              <span className="hidden sm:block text-sm text-gray-300">{user?.full_name || user?.email}</span>
              <ChevronDown className="w-4 h-4 text-gray-500" />
            </button>
            {open && (
              <div className="absolute right-0 mt-1 w-44 bg-gray-800 rounded-lg shadow-xl border border-gray-700 py-1 z-50">
                <button onClick={() => { logout(); navigate('/login') }}
                  className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-400 hover:bg-gray-700">
                  <LogOut className="w-4 h-4" />Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}
