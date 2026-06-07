export function formatDistanceToNow(dateStr) {
  const diff = Math.floor((Date.now() - new Date(dateStr)) / 1000)
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`
  if (diff < 86400) return `${Math.floor(diff/3600)}h ago`
  return `${Math.floor(diff/86400)}d ago`
}

export function riskColor(level) {
  return { HIGH: 'text-red-400', MODERATE: 'text-yellow-400', LOW: 'text-blue-400', MINIMAL: 'text-green-400' }[level] || 'text-gray-400'
}

export function riskBg(level) {
  return { HIGH: 'bg-red-900/30 border-red-700', MODERATE: 'bg-yellow-900/30 border-yellow-700', LOW: 'bg-blue-900/30 border-blue-700', MINIMAL: 'bg-green-900/30 border-green-700' }[level] || 'bg-gray-800 border-gray-700'
}
