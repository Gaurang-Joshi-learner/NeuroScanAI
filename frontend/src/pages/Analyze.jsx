import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../lib/api'
import { Upload, FileText, X, AlertCircle, Loader, Activity, Mic, CheckCircle } from 'lucide-react'

const TYPES = [
  {
    id: 'seizure', label: 'Seizure Detection', icon: Activity, color: 'red',
    desc: 'Upload a CHB-MIT format .edf file. Detects seizure events and generates a risk timeline.',
    accept: '.edf', hint: 'Accepts .edf files (European Data Format)',
  },
  {
    id: 'speech', label: 'Speech Decoding', icon: Mic, color: 'violet',
    desc: 'Upload a KaraOne format .mat file. Decodes imagined speech across 11 phoneme/word classes.',
    accept: '.mat', hint: 'Accepts .mat files (MATLAB/KaraOne format)',
  },
]

export default function Analyze() {
  const navigate = useNavigate()
  const inputRef = useRef()
  const [type, setType] = useState('seizure')
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')

  const selected = TYPES.find(t => t.id === type)

  const handleFile = f => {
    if (!f) return
    const ext = f.name.split('.').pop().toLowerCase()
    const allowed = type === 'seizure' ? ['edf'] : ['mat']
    if (!allowed.includes(ext)) { setError(`For ${selected.label}, upload a .${allowed[0]} file`); return }
    if (f.size > 100 * 1024 * 1024) { setError('File must be under 100MB'); return }
    setError(''); setFile(f)
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true); setProgress(0); setError('')
    const form = new FormData()
    form.append('file', file)
    try {
      const { data } = await api.post(`/analysis/${type}`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: e => setProgress(Math.round(e.loaded / e.total * 100)),
      })
      navigate(`/results/${data.id}`)
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Is the backend running?')
      setUploading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">New EEG Analysis</h1>
        <p className="text-gray-400 mt-1">Upload an EEG recording for AI-powered analysis</p>
      </div>

      {/* Analysis type selector */}
      <div className="grid grid-cols-2 gap-3 mb-6">
        {TYPES.map(t => (
          <button key={t.id} onClick={() => { setType(t.id); setFile(null); setError('') }}
            className={`p-4 rounded-xl border-2 text-left transition-all ${
              type === t.id
                ? t.id === 'seizure' ? 'border-red-500 bg-red-900/20' : 'border-violet-500 bg-violet-900/20'
                : 'border-gray-700 bg-gray-900 hover:border-gray-600'
            }`}>
            <t.icon className={`w-6 h-6 mb-2 ${type === t.id ? (t.id === 'seizure' ? 'text-red-400' : 'text-violet-400') : 'text-gray-500'}`} />
            <p className={`font-medium text-sm ${type === t.id ? 'text-white' : 'text-gray-400'}`}>{t.label}</p>
            <p className="text-xs text-gray-500 mt-1">{t.hint}</p>
          </button>
        ))}
      </div>

      {/* Description */}
      <div className="card mb-6 bg-gray-800/50 border-gray-700">
        <p className="text-sm text-gray-300">{selected.desc}</p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}
        onClick={() => !file && inputRef.current?.click()}
        className={`card border-2 border-dashed transition-all text-center py-12 cursor-pointer
          ${dragOver ? 'border-violet-500 bg-violet-900/10' : 'border-gray-700 hover:border-gray-600'}
          ${file ? 'cursor-default' : ''}`}
      >
        <input ref={inputRef} type="file" accept={selected.accept} className="hidden"
          onChange={e => handleFile(e.target.files[0])} />
        {!file ? (
          <>
            <div className="w-16 h-16 bg-gray-800 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <Upload className="w-12 h-12 text-violet-400" />
            </div>
            <p className="text-gray-300 font-medium">Drop your {selected.accept} file here</p>
            <p className="text-gray-500 text-sm mt-1">or click to browse · Max 100MB</p>
          </>
        ) : (
          <div className="flex items-center justify-center gap-4">
            <div className="w-12 h-12 bg-violet-900/30 rounded-xl flex items-center justify-center">
              <FileText className="w-6 h-6 text-violet-400" />
            </div>
            <div className="text-left">
              <p className="font-medium text-white">{file.name}</p>
              <p className="text-sm text-gray-400">{(file.size/1024/1024).toFixed(2)} MB</p>
            </div>
            <button onClick={e => { e.stopPropagation(); setFile(null) }}
              className="p-1.5 hover:bg-gray-700 rounded-lg text-gray-400 ml-2">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 mt-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />{error}
        </div>
      )}

      {uploading && (
        <div className="mt-4">
          <div className="flex justify-between text-sm text-gray-400 mb-1">
            <span>Uploading...</span><span>{progress}%</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div className="bg-violet-500 h-2 rounded-full transition-all" style={{ width: `${progress}%` }} />
          </div>
          {progress === 100 && (
            <p className="text-sm text-violet-400 mt-2 flex items-center gap-1">
              <Loader className="w-3.5 h-3.5 animate-spin" />Running EEG analysis pipeline...
            </p>
          )}
        </div>
      )}

      {/* What happens info */}
      <div className="card mt-6 border-gray-700 bg-gray-900/50">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">Pipeline steps</p>
        <div className="space-y-2">
          {(type === 'seizure' ? [
            'EDF loaded → bandpass 0.5–40Hz + notch 60Hz filter',
            '4-second epochs with 50% overlap extracted',
            'Band power + Hjorth features computed per channel',
            'EEGNet CNN + XGBoost ensemble inference',
            'Per-epoch seizure probability timeline generated',
          ] : [
            'MAT file loaded → bandpass 1–40Hz filter',
            'Downsampled to 256Hz → per-trial features extracted',
            'PCA dimensionality reduction applied',
            'EEGNet + XGBoost ensemble predicts phoneme/word',
            'Per-trial confidence scores returned',
          ]).map((step, i) => (
            <div key={i} className="flex items-start gap-2 text-sm text-gray-400">
              <CheckCircle className="w-4 h-4 text-violet-500 mt-0.5 flex-shrink-0" />{step}
            </div>
          ))}
        </div>
      </div>

      <button onClick={handleUpload} disabled={!file || uploading}
        className="btn-primary w-full mt-6 py-3 flex items-center justify-center gap-2 text-base">
        {uploading ? <><Loader className="w-5 h-5 animate-spin" />Analysing...</>
                   : <><Upload className="w-5 h-5" />Run {selected.label}</>}
      </button>

      <p className="text-xs text-gray-600 text-center mt-4">
        For research use only. Not a clinical diagnostic tool.
      </p>
    </div>
  )
}
