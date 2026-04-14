import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { flakeBlends, solidColors } from './data/palette'

type Point = { x: number; y: number }

type FlakeDot = { x: number; y: number; r: number; c: number }

type QuoteRequest = {
  customerName: string
  phone: string
  email: string
  address: string
  squareFeet: string
  notes: string
}

const flakeToneMap: Record<string, string[]> = {
  // tuned per latest direction
  Stonewash: ['#b5c8e6', '#6186bd', '#1f4578'], // same profile as previous Orbit
  Nightfall: ['#9ca3af', '#4b5563', '#1f2937'], // more charcoal + black
  Orbit: ['#1f4578', '#1f4578', '#2b5b99', '#111111', '#f4f4f4', '#c9ced6'], // ~50% blue bias + black/white/light grey
  Outback: ['#e8dccf', '#8b6a4d', '#2b2219'], // more brown + black
  'Cabin Fever': ['#f4d36b', '#f5f5f5', '#1f1f1f'], // yellow + white + black
  'Tan Blend': ['#e0c8a2', '#b98e61', '#7c5d3f'],
  'Tidal Wave': ['#d9dde5', '#8fa5b8', '#4b6478'],
  'Stony Creek': ['#d5d7d8', '#9fa4a6', '#5f666b'],
}

const solidBaseMap: Record<string, string> = {
  Grey: '#7d8289',
  Tan: '#a88c6e',
  Charcoal: '#454850',
}

function seededDots(seed: string, count = 650, toneCount = 3): FlakeDot[] {
  let h = 2166136261
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  const rand = () => {
    h += 0x6d2b79f5
    let t = Math.imul(h ^ (h >>> 15), 1 | h)
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  const dots: FlakeDot[] = []
  for (let i = 0; i < count; i++) {
    dots.push({
      x: rand() * 100,
      y: rand() * 100,
      r: 0.12 + rand() * 0.28,
      c: Math.floor(rand() * toneCount),
    })
  }
  return dots
}

const QUOTE_WEBHOOK_URL = (import.meta.env.VITE_QUOTE_WEBHOOK_URL as string | undefined) || '/api/quote'
const BRAND_TONE_1 = '#F6CCAA'
const BRAND_TONE_2 = '#3FCBA2'
const LOGO_PATH = '/logo-icon.png'

function App() {
  const [selectedSolid, setSelectedSolid] = useState(solidColors[0])
  const [selectedFlake, setSelectedFlake] = useState<string>('Stonewash')
  const [imageUrl, setImageUrl] = useState<string>('')
  const [maskPoints, setMaskPoints] = useState<Point[]>([])
  const [maskMode, setMaskMode] = useState<'manual' | 'auto'>('manual')
  const [autoMaskStatus, setAutoMaskStatus] = useState('')
  const [autoMaskLoading, setAutoMaskLoading] = useState(false)
  const [compare, setCompare] = useState(100)
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [projectName] = useState('')
  const [companyName] = useState('LuxShield Coatings')
  const [companyContact] = useState('717-490-2643 • Sales@LuxShiledCoatings.com • LuxShieldCoatings.com')
  const [quote, setQuote] = useState<QuoteRequest>({
    customerName: '',
    phone: '',
    email: '',
    address: '',
    squareFeet: '',
    notes: '',
  })
  const [submittingQuote, setSubmittingQuote] = useState(false)
  const [quoteStatus, setQuoteStatus] = useState('')
  const [isMobilePreview, setIsMobilePreview] = useState(false)
  const compareRef = useRef<HTMLDivElement | null>(null)

  const isQuoteReady = useMemo(() => {
    const hasName = quote.customerName.trim().length > 0
    const hasContact = quote.phone.trim().length > 0 || quote.email.trim().length > 0
    return Boolean(imageUrl && hasName && hasContact)
  }, [imageUrl, quote.customerName, quote.phone, quote.email])

  const finishLabel = useMemo(() => {
    return selectedFlake === 'None'
      ? `Solid: ${selectedSolid} (No Flake)`
      : `Solid: ${selectedSolid} + Flake: ${selectedFlake}`
  }, [selectedSolid, selectedFlake])

  const polygonPointsAttr = useMemo(() => maskPoints.map((p) => `${p.x},${p.y}`).join(' '), [maskPoints])

  const insetPolygonPointsAttr = useMemo(() => {
    if (maskPoints.length < 3) return ''

    const cx = maskPoints.reduce((sum, p) => sum + p.x, 0) / maskPoints.length
    const cy = maskPoints.reduce((sum, p) => sum + p.y, 0) / maskPoints.length
    const insetAmount = 0.8

    return maskPoints
      .map((p) => {
        const dx = cx - p.x
        const dy = cy - p.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 0.01) return `${p.x},${p.y}`
        const ratio = insetAmount / dist
        return `${p.x + dx * ratio},${p.y + dy * ratio}`
      })
      .join(' ')
  }, [maskPoints])

  const flakeTones = flakeToneMap[selectedFlake] ?? ['#d0d4db', '#8f96a4', '#59606f']
  const baseSolid = solidBaseMap[selectedSolid] ?? '#7d8289'

  const ultraFlakeMacroDots = useMemo(
    () => seededDots(`${selectedSolid}-${selectedFlake}-ultra-macro`, 22000, flakeTones.length),
    [selectedSolid, selectedFlake, flakeTones.length],
  )
  const ultraFlakeMicroDots = useMemo(
    () => seededDots(`${selectedSolid}-${selectedFlake}-ultra-micro`, 70000, flakeTones.length),
    [selectedSolid, selectedFlake, flakeTones.length],
  )

  const liveFlakeMacroDots = useMemo(
    () => seededDots(`${selectedSolid}-${selectedFlake}-live-macro`, isMobilePreview ? 8000 : 22000, flakeTones.length),
    [selectedSolid, selectedFlake, flakeTones.length, isMobilePreview],
  )
  const liveFlakeMicroDots = useMemo(
    () => seededDots(`${selectedSolid}-${selectedFlake}-live-micro`, isMobilePreview ? 25000 : 70000, flakeTones.length),
    [selectedSolid, selectedFlake, flakeTones.length, isMobilePreview],
  )

  useEffect(() => {
    if (typeof window === 'undefined') return
    const mq = window.matchMedia('(max-width: 768px)')
    const apply = () => setIsMobilePreview(mq.matches)
    apply()
    mq.addEventListener?.('change', apply)
    return () => mq.removeEventListener?.('change', apply)
  }, [])

  // Keep segmentation model warm — ping on load then every 60s while page is open
  useEffect(() => {
    const ping = () => fetch('/api/warmup').catch(() => {})
    const t = setTimeout(ping, 3000)
    const iv = setInterval(ping, 60_000)
    return () => { clearTimeout(t); clearInterval(iv) }
  }, [])

  function toNormalizedPoint(clientX: number, clientY: number, rect: DOMRect) {
    const x = ((clientX - rect.left) / rect.width) * 100
    const y = ((clientY - rect.top) / rect.height) * 100
    return {
      x: Math.max(0, Math.min(100, x)),
      y: Math.max(0, Math.min(100, y)),
    }
  }

  function addMaskPoint(clientX: number, clientY: number, rect: DOMRect) {
    const p = toNormalizedPoint(clientX, clientY, rect)
    setMaskPoints((pts) => [...pts, p])
  }

  function updateMaskPoint(index: number, clientX: number, clientY: number, rect: DOMRect) {
    const p = toNormalizedPoint(clientX, clientY, rect)
    setMaskPoints((pts) => pts.map((pt, i) => (i === index ? p : pt)))
  }


  async function autoDetectFloorMask() {
    if (!imageUrl) return

    const startTime = Date.now()
    let timerInterval: ReturnType<typeof setInterval> | undefined
    try {
      setAutoMaskLoading(true)
      timerInterval = setInterval(() => {
        const elapsed = Math.round((Date.now() - startTime) / 1000)
        setAutoMaskStatus(`Segmenting floor with AI… ${elapsed}s elapsed`)
      }, 1000)
      setAutoMaskStatus('Segmenting floor with AI… 0s elapsed')

      // Downscale image to keep payload small (max ~800px wide)
      const img = new Image()
      img.src = imageUrl
      await img.decode()

      const maxW = 800
      const scale = img.naturalWidth > maxW ? maxW / img.naturalWidth : 1
      const w = Math.round(img.naturalWidth * scale)
      const h = Math.round(img.naturalHeight * scale)

      const c = document.createElement('canvas')
      c.width = w
      c.height = h
      const ctx = c.getContext('2d')
      if (!ctx) throw new Error('Canvas unavailable')
      ctx.drawImage(img, 0, 0, w, h)

      const imageDataUrl = c.toDataURL('image/jpeg', 0.75)

      const res = await fetch('/api/auto-mask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageDataUrl }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error((err as { error?: string }).error || `API error ${res.status}`)
      }

      const data = (await res.json()) as { points: { x: number; y: number }[]; error?: string }

      if (data.error) {
        throw new Error(data.error)
      }

      if (!data.points || data.points.length < 3) {
        throw new Error('Could not detect floor boundary')
      }

      clearInterval(timerInterval)
      const totalTime = Math.round((Date.now() - startTime) / 1000)
      setMaskPoints(data.points)
      setMaskMode('auto')
      setAutoMaskStatus(`AI floor mask applied (${data.points.length} points) in ${totalTime}s. Drag points to fine-tune.`)
    } catch (err) {
      clearInterval(timerInterval)
      console.error('Auto-mask failed:', err)
      setAutoMaskStatus(`AI detection failed: ${(err as Error).message}. Use manual points for now.`)
    } finally {
      setAutoMaskLoading(false)
    }
  }


  function drawAfterOverlay(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    microDots: FlakeDot[],
    macroDots: FlakeDot[],
  ) {
    if (maskPoints.length < 3) return

    const pts = maskPoints.map((p) => ({ x: (p.x / 100) * w, y: (p.y / 100) * h }))

    ctx.save()
    ctx.beginPath()
    ctx.moveTo(pts[0].x, pts[0].y)
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
    ctx.closePath()
    ctx.clip()

    // Solid base with slight transparency so floor shading/details still read naturally
    ctx.globalAlpha = 1
    ctx.fillStyle = baseSolid
    ctx.fillRect(0, 0, w, h)
    ctx.globalAlpha = 1

    // Dense flake broadcast
    if (selectedFlake !== 'None') {
      for (const d of microDots) {
        ctx.globalAlpha = 0.72
        ctx.fillStyle = flakeTones[d.c]
        ctx.beginPath()
        ctx.arc((d.x / 100) * w, (d.y / 100) * h, Math.max(0.6, d.r * 0.25 * (w / 100)), 0, Math.PI * 2)
        ctx.fill()
      }
      for (const d of macroDots) {
        ctx.globalAlpha = 0.98
        ctx.fillStyle = flakeTones[d.c]
        ctx.beginPath()
        ctx.arc((d.x / 100) * w, (d.y / 100) * h, Math.max(0.9, d.r * 0.41 * (w / 100)), 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Subtle texture so the floor reads as coated material (not a flat block)
    for (let i = 0; i < 1200; i++) {
      const x = Math.random() * w
      const y = Math.random() * h
      const shade = Math.random() > 0.5 ? 255 : 0
      const alpha = 0.02 + Math.random() * 0.03
      ctx.fillStyle = `rgba(${shade},${shade},${shade},${alpha})`
      ctx.fillRect(x, y, 1, 1)
    }

    // Gloss (reduced)
    const g1 = ctx.createLinearGradient(0, 0, w, h)
    g1.addColorStop(0, 'rgba(255,255,255,0.30)')
    g1.addColorStop(0.32, 'rgba(255,255,255,0.10)')
    g1.addColorStop(1, 'rgba(255,255,255,0.0)')
    ctx.globalAlpha = 0.58
    ctx.fillStyle = g1
    ctx.fillRect(0, 0, w, h)

    const g2 = ctx.createRadialGradient(w * 0.18, h * 0.16, 0, w * 0.18, h * 0.16, Math.max(w, h) * 0.55)
    g2.addColorStop(0, 'rgba(255,255,255,0.26)')
    g2.addColorStop(0.45, 'rgba(255,255,255,0.07)')
    g2.addColorStop(1, 'rgba(255,255,255,0.0)')
    ctx.globalAlpha = 0.42
    ctx.fillStyle = g2
    ctx.fillRect(0, 0, w, h)

    ctx.restore()
    ctx.globalAlpha = 1
  }

  async function buildAfterPreviewDataUrl(maxWidth = 1200) {
    const img = new Image()
    img.src = imageUrl
    await img.decode()

    const scale = img.naturalWidth > maxWidth ? maxWidth / img.naturalWidth : 1
    const w = Math.max(1, Math.round(img.naturalWidth * scale))
    const h = Math.max(1, Math.round(img.naturalHeight * scale))

    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas not supported')

    ctx.drawImage(img, 0, 0, w, h)
    drawAfterOverlay(ctx, w, h, ultraFlakeMicroDots, ultraFlakeMacroDots)

    return canvas.toDataURL('image/jpeg', 0.9)
  }

  async function submitQuoteRequest() {
    if (!imageUrl) {
      alert('Upload a garage photo first.')
      return
    }
    if (!quote.customerName.trim() || (!quote.phone.trim() && !quote.email.trim())) {
      alert('Add customer name and at least phone or email.')
      return
    }

    setSubmittingQuote(true)
    setQuoteStatus('Submitting quote request...')

    try {
      const afterPreview = await buildAfterPreviewDataUrl(1200)
      const payload = {
        submittedAt: new Date().toISOString(),
        projectName: projectName.trim() || 'Epoxy Flooring Proposal',
        finishLabel,
        selectedSolid,
        selectedFlake,
        compare,
        customer: quote,
        company: {
          companyName,
          companyContact,
        },
        assets: {
          afterPreviewDataUrl: afterPreview,
        },
      }

      const res = await fetch(QUOTE_WEBHOOK_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!res.ok) throw new Error(`Request failed (${res.status})`)

      setQuoteStatus('Quote request sent successfully.')
      setQuote((prev) => ({ ...prev, notes: '' }))
    } catch {
      setQuoteStatus('Quote submit failed. Try again or export quote card and send manually.')
    } finally {
      setSubmittingQuote(false)
    }
  }

  async function saveDataUrlWithFallback(dataUrl: string, filename: string) {
    const isFacebookInApp = /FBAN|FBAV|FB_IAB|Instagram/i.test(navigator.userAgent)

    try {
      const res = await fetch(dataUrl)
      const blob = await res.blob()
      const file = new File([blob], filename, { type: 'image/png' })
      if (navigator.share && navigator.canShare?.({ files: [file] })) {
        await navigator.share({ files: [file], title: 'Epoxy Preview' })
        return
      }
    } catch {
      // continue to download fallback
    }

    try {
      const a = document.createElement('a')
      a.href = dataUrl
      a.download = filename
      document.body.appendChild(a)
      a.click()
      a.remove()
      return
    } catch {
      // continue to open-tab fallback
    }

    // Some in-app browsers (especially Facebook/Instagram) block downloads.
    if (isFacebookInApp) {
      const w = window.open(dataUrl, '_blank')
      if (!w) window.location.href = dataUrl
      alert('Facebook in-app browser may block direct download. Long-press the opened image and save it, or open this page in your system browser.')
      return
    }

    const w = window.open(dataUrl, '_blank')
    if (!w) window.location.href = dataUrl
  }

  async function exportQuoteCard() {
    try {
      if (!imageUrl) {
        alert('Upload a photo first.')
        return
      }

      const img = new Image()
      img.src = imageUrl
      await img.decode()

      const w = img.naturalWidth
      const h = img.naturalHeight

      const afterCanvas = document.createElement('canvas')
      afterCanvas.width = w
      afterCanvas.height = h
      const afterCtx = afterCanvas.getContext('2d')
      if (!afterCtx) return
      afterCtx.drawImage(img, 0, 0, w, h)
      drawAfterOverlay(afterCtx, w, h, ultraFlakeMicroDots, ultraFlakeMacroDots)

      const outW = 1600
      const headerH = 240
      const bodyH = 900
      const outH = headerH + bodyH + 180

      const canvas = document.createElement('canvas')
      canvas.width = outW
      canvas.height = outH
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      ctx.fillStyle = '#000000'
      ctx.fillRect(0, 0, outW, outH)

      // Subtle textured black background
      for (let i = 0; i < 2200; i++) {
        const x = Math.random() * outW
        const y = Math.random() * outH
        const a = 0.02 + Math.random() * 0.03
        ctx.fillStyle = `rgba(255,255,255,${a})`
        ctx.fillRect(x, y, 1, 1)
      }

      // Modern abstract line accents
      ctx.save()
      ctx.globalAlpha = 0.32
      ctx.strokeStyle = BRAND_TONE_2
      ctx.lineCap = 'round'

      const drawLine = (x1: number, y1: number, x2: number, y2: number, width: number) => {
        ctx.beginPath()
        ctx.lineWidth = width
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
      }

      drawLine(outW * 0.72, outH * 0.08, outW * 0.95, outH * 0.08, 3)
      drawLine(outW * 0.76, outH * 0.11, outW * 0.95, outH * 0.11, 2)
      drawLine(outW * 0.80, outH * 0.14, outW * 0.95, outH * 0.14, 1.5)

      drawLine(outW * 0.06, outH * 0.88, outW * 0.28, outH * 0.88, 3)
      drawLine(outW * 0.06, outH * 0.91, outW * 0.24, outH * 0.91, 2)
      drawLine(outW * 0.06, outH * 0.94, outW * 0.20, outH * 0.94, 1.5)

      ctx.restore()

      const brandGrad = ctx.createLinearGradient(0, 0, outW, 0)
      brandGrad.addColorStop(0, BRAND_TONE_1)
      brandGrad.addColorStop(1, BRAND_TONE_2)
      ctx.fillStyle = brandGrad
      ctx.fillRect(0, 0, outW, 12)

      ctx.fillStyle = BRAND_TONE_1
      ctx.font = 'bold 54px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif'
      ctx.fillText(companyName || 'Your Company', 64, 90)

      try {
        const logo = new Image()
        logo.src = LOGO_PATH
        await logo.decode()
        const targetH = 92
        const targetW = (logo.naturalWidth / logo.naturalHeight) * targetH
        const x = outW - 64 - targetW
        const y = 32
        ctx.drawImage(logo, x, y, targetW, targetH)
      } catch {
        // If logo is missing/unreadable, continue export without it.
      }

      ctx.font = '30px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif'
      ctx.fillStyle = BRAND_TONE_1
      ctx.fillText(companyContact || 'Phone • Email', 64, 136)

      const label = projectName.trim() || 'Epoxy Flooring Proposal'
      const dateLabel = new Date().toLocaleDateString()
      ctx.font = '28px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif'
      ctx.fillStyle = BRAND_TONE_1
      ctx.fillText(`Project: ${label}`, 64, 182)
      ctx.fillText(`Date: ${dateLabel}`, 64, 218)

      const margin = 64
      const gap = 36
      const panelW = Math.floor((outW - margin * 2 - gap) / 2)
      const panelH = bodyH - 110
      const topY = headerH + 40

      ctx.fillStyle = '#111827'
      ctx.fillRect(margin, topY, panelW, panelH)
      ctx.fillRect(margin + panelW + gap, topY, panelW, panelH)

      ctx.drawImage(img, margin, topY, panelW, panelH)
      ctx.drawImage(afterCanvas, margin + panelW + gap, topY, panelW, panelH)

      ctx.font = 'bold 30px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif'
      ctx.fillStyle = BRAND_TONE_1
      ctx.fillText('Before', margin, topY - 10)
      ctx.fillText('After', margin + panelW + gap, topY - 10)

      ctx.font = '27px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif'
      ctx.fillStyle = BRAND_TONE_1
      ctx.fillText(`Finish: ${finishLabel}`, margin, outH - 98)
      ctx.fillText('System: Decorative flake epoxy with high-gloss topcoat', margin, outH - 62)

      ctx.font = '20px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif'
      ctx.fillStyle = BRAND_TONE_1
      ctx.fillText('Disclaimer: This is a digital preview for visualization only. Final color/texture may vary due to lighting, surface prep,', margin, outH - 30)
      ctx.fillText('application method, and monitor/device settings.', margin, outH - 8)

      const filename = `epoxy-quote-${Date.now()}.png`
      const dataUrl = canvas.toDataURL('image/png')
      await saveDataUrlWithFallback(dataUrl, filename)
    } catch {
      alert('Quote export failed on this device. Try again or use comparison export.')
    }
  }

  async function exportPreviewImage(kind: 'comparison' | 'after') {
    try {
      if (!imageUrl) return

      const img = new Image()
      img.src = imageUrl
      await img.decode()

      const w = img.naturalWidth
      const h = img.naturalHeight

      const afterCanvas = document.createElement('canvas')
      afterCanvas.width = w
      afterCanvas.height = h
      const afterCtx = afterCanvas.getContext('2d')
      if (!afterCtx) return
      afterCtx.drawImage(img, 0, 0, w, h)
      drawAfterOverlay(afterCtx, w, h, ultraFlakeMicroDots, ultraFlakeMacroDots)

      const canvas = document.createElement('canvas')
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      if (kind === 'after') {
        ctx.drawImage(afterCanvas, 0, 0)
      } else {
        ctx.drawImage(img, 0, 0, w, h)
        const cut = Math.floor((compare / 100) * w)
        if (cut > 0) {
          ctx.drawImage(afterCanvas, 0, 0, cut, h, 0, 0, cut, h)
          ctx.fillStyle = '#ffffff'
          ctx.fillRect(Math.max(0, cut - 1), 0, 2, h)
        }
      }

      const filename = `epoxy-${kind}-${Date.now()}.png`
      const dataUrl = canvas.toDataURL('image/png')
      await saveDataUrlWithFallback(dataUrl, filename)
    } catch {
      alert('Save/share failed on this device. Please screenshot for now, and I will patch further.')
    }
  }

  return (
    <main className="wrap">
      <header>
        <h1>Epoxy Visualizer V4 • Grounded SAM</h1>
        <p>Mobile-first before/after preview builder for garage flooring.</p>
      </header>

      <section className="card">
        <h2>1) Upload Garage Photo</h2>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (!file) return
            setImageUrl(URL.createObjectURL(file))
            setMaskPoints([])
            setMaskMode('manual')
            setAutoMaskStatus('')
          }}
        />
        {!imageUrl && <p className="muted">No image yet.</p>}
      </section>

      <section className="card">
        <h2>2) Floor Mask Tool (Polygon)</h2>
        <p className="muted">Tap points around the floor boundary. Use clear/undo to adjust.</p>
        {imageUrl ? (
          <>
            <div className="row">
              <button onClick={() => void autoDetectFloorMask()} disabled={autoMaskLoading}>
                {autoMaskLoading ? 'Detecting…' : 'Auto Detect Floor (Beta)'}
              </button>
              <button
                onClick={() => {
                  setMaskPoints((pts) => pts.slice(0, -1))
                  setMaskMode('manual')
                }}
                disabled={!maskPoints.length}
              >
                Undo point
              </button>
              <button
                onClick={() => {
                  setMaskPoints([])
                  setMaskMode('manual')
                }}
                disabled={!maskPoints.length}
              >
                Clear mask
              </button>
            </div>
            <div
              className="stage"
              onPointerDown={(e) => {
                const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect()

                if (dragIndex !== null) return
                const target = e.target as HTMLElement
                if (target.dataset.point === 'mask-dot') return
                addMaskPoint(e.clientX, e.clientY, rect)
                setMaskMode('manual')
              }}
              onPointerMove={(e) => {
                const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect()

                if (dragIndex === null) return
                updateMaskPoint(dragIndex, e.clientX, e.clientY, rect)
                setMaskMode('manual')
              }}
              onPointerUp={() => {
                setDragIndex(null)
              }}
              onPointerLeave={() => {
                setDragIndex(null)
              }}
            >
              <img src={imageUrl} alt="garage" className="preview" />
              {autoMaskLoading && (
                <div className="auto-mask-overlay">
                  <div className="auto-mask-spinner" />
                  <p>{autoMaskStatus}</p>
                </div>
              )}
              <svg className="mask-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                {maskPoints.length >= 3 && <polygon className="mask-fill" points={polygonPointsAttr} />}
                {maskPoints.length >= 2 && <polyline className="mask-line" fill="none" points={polygonPointsAttr} />}
                {maskPoints.map((p, idx) => (
                  <circle
                    key={`${p.x}-${p.y}-${idx}`}
                    cx={p.x}
                    cy={p.y}
                    r={0.8}
                    className="mask-dot"
                    data-point="mask-dot"
                    onPointerDown={(e) => {
                      e.stopPropagation()
                      setDragIndex(idx)
                    }}
                  />
                ))}
              </svg>
            </div>
            <small>{maskPoints.length} point(s) selected • Mode: {maskMode === 'auto' ? 'Auto' : 'Manual'}</small>
            {autoMaskStatus && <small className="muted">{autoMaskStatus}</small>}
          </>
        ) : (
          <p className="muted">Upload a photo first.</p>
        )}
      </section>

      <section className="card">
        <h2>3) Finish Selection</h2>

        <div className="row two-col" style={{ marginTop: 10 }}>
          <label className="field-inline">
            <span>Solid Base (Opaque)</span>
            <select value={selectedSolid} onChange={(e) => setSelectedSolid(e.target.value)}>
              {solidColors.map((c) => <option key={c}>{c}</option>)}
            </select>
          </label>
          <label className="field-inline">
            <span>Flake Blend</span>
            <select value={selectedFlake} onChange={(e) => setSelectedFlake(e.target.value)}>
              <option value="None">None</option>
              {flakeBlends.map((f) => <option key={f}>{f}</option>)}
            </select>
          </label>
        </div>

        <p className="badge">Selected finish: {finishLabel} • High-gloss</p>
      </section>

      <section className="card">
        <h2>4) Before/After Slider + Gloss Preview</h2>
        <div className="row" style={{ marginTop: 8 }}>
          <span className="muted">Preview/export quality: Ultra.</span>
        </div>
        {imageUrl ? (
          <>
            <div className="row">
              <label className="slider-label">Before / After: {compare}%</label>
              <input type="range" min={0} max={100} value={compare} onChange={(e) => setCompare(Number(e.target.value))} />
              <input
                type="number"
                min={0}
                max={100}
                value={compare}
                onChange={(e) => {
                  const v = Number(e.target.value)
                  setCompare(Number.isFinite(v) ? Math.max(0, Math.min(100, v)) : 0)
                }}
                style={{ width: 86 }}
              />
              <button onClick={() => exportPreviewImage('comparison')}>Save/Share Comparison</button>
              <button onClick={() => exportPreviewImage('after')}>Save/Share After</button>
            </div>

            <div className="stage compare-stage" ref={compareRef}>
              <img src={imageUrl} alt="garage before" className="preview" />

              <div className="after-layer" style={{ clipPath: `inset(0 ${100 - compare}% 0 0)` }}>
                <img src={imageUrl} alt="garage after" className="preview after-preview" />
                <svg className="mask-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                  <defs>
                    <clipPath id="floorClip">
                      {maskPoints.length >= 3 && <polygon points={insetPolygonPointsAttr} />}
                    </clipPath>
                    <linearGradient id="glossGradient" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor="rgba(255,255,255,0.52)" />
                      <stop offset="32%" stopColor="rgba(255,255,255,0.18)" />
                      <stop offset="100%" stopColor="rgba(255,255,255,0.0)" />
                    </linearGradient>
                    <radialGradient id="glossSpot" cx="18%" cy="16%" r="55%">
                      <stop offset="0%" stopColor="rgba(255,255,255,0.42)" />
                      <stop offset="45%" stopColor="rgba(255,255,255,0.10)" />
                      <stop offset="100%" stopColor="rgba(255,255,255,0.0)" />
                    </radialGradient>
                  </defs>

                  {maskPoints.length >= 3 && <polygon className="overlay-base" points={polygonPointsAttr} fill={baseSolid} />}

                  {maskPoints.length >= 3 && selectedFlake !== 'None' && (
                    <g clipPath="url(#floorClip)">
                      {liveFlakeMicroDots.map((d, i) => (
                        <circle key={`micro-${i}`} cx={d.x} cy={d.y} r={Math.max(0.05, d.r * 0.25)} fill={flakeTones[d.c]} opacity={0.72} />
                      ))}
                      {liveFlakeMacroDots.map((d, i) => (
                        <circle key={`macro-${i}`} cx={d.x} cy={d.y} r={Math.max(0.08, d.r * 0.41)} fill={flakeTones[d.c]} opacity={0.98} />
                      ))}
                    </g>
                  )}

                  {maskPoints.length >= 3 && <polygon className="overlay-gloss" points={polygonPointsAttr} fill="url(#glossGradient)" />}
                  {maskPoints.length >= 3 && <polygon className="overlay-gloss-spot" points={polygonPointsAttr} fill="url(#glossSpot)" />}
                </svg>
              </div>

              <div className="compare-handle" style={{ left: `${compare}%` }} />
            </div>
          </>
        ) : (
          <p className="muted">Upload image to preview.</p>
        )}
      </section>

      <section className="card">
        <h2>5) Client Quote Request Form</h2>
        <p className="muted">Use this for website leads: visualize first, then submit for a formal quote follow-up.</p>
        <div className="row two-col" style={{ marginTop: 10 }}>
          <label className="field-inline">
            <span>Customer Name *</span>
            <input value={quote.customerName} onChange={(e) => setQuote((q) => ({ ...q, customerName: e.target.value }))} />
          </label>
          <label className="field-inline">
            <span>Phone</span>
            <input value={quote.phone} onChange={(e) => setQuote((q) => ({ ...q, phone: e.target.value }))} />
          </label>
        </div>
        <div className="row two-col">
          <label className="field-inline">
            <span>Email</span>
            <input value={quote.email} onChange={(e) => setQuote((q) => ({ ...q, email: e.target.value }))} />
          </label>
          <label className="field-inline">
            <span>Estimated Square Feet</span>
            <input value={quote.squareFeet} onChange={(e) => setQuote((q) => ({ ...q, squareFeet: e.target.value }))} />
          </label>
        </div>
        <div className="row">
          <label className="field-inline" style={{ width: '100%' }}>
            <span>Project Address</span>
            <input value={quote.address} onChange={(e) => setQuote((q) => ({ ...q, address: e.target.value }))} />
          </label>
        </div>
        <div className="row">
          <label className="field-inline" style={{ width: '100%' }}>
            <span>Notes</span>
            <textarea rows={4} value={quote.notes} onChange={(e) => setQuote((q) => ({ ...q, notes: e.target.value }))} />
          </label>
        </div>
        <div className="row">
          <button onClick={submitQuoteRequest} disabled={submittingQuote}>{submittingQuote ? 'Submitting...' : 'Submit Quote Request'}</button>
          <span className="muted">Endpoint: {QUOTE_WEBHOOK_URL}</span>
        </div>
        {quoteStatus && <p className="status-msg">{quoteStatus}</p>}
      </section>

      <section className="card">
        <h2>6) Quote Export</h2>
        <div className="row two-col" style={{ marginTop: 10 }}>
          <label className="field-inline">
            <span>Company Name</span>
            <input value={companyName} readOnly />
          </label>
          <label className="field-inline">
            <span>Contact (phone/email)</span>
            <input value={companyContact} readOnly />
          </label>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={exportQuoteCard} disabled={!isQuoteReady}>Export Digital Visualization</button>
          {!isQuoteReady && <span className="muted">Fill #5 (name + phone/email) and upload a photo first.</span>}
        </div>
      </section>

      
    </main>
  )
}

export default App
