import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function AuthCallback() {
  const { loginWithTokens } = useAuth()
  const navigate = useNavigate()
  useEffect(() => {
    const params = new URLSearchParams(window.location.hash.substring(1))
    const access = params.get('access_token')
    const refresh = params.get('refresh_token')
    if (access && refresh) { loginWithTokens(access, refresh); navigate('/dashboard', { replace: true }) }
    else navigate('/login', { replace: true })
  }, [])
  return <div className="min-h-screen flex items-center justify-center"><div className="w-8 h-8 border-4 border-violet-500 border-t-transparent rounded-full animate-spin" /></div>
}
