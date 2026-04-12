type QuotePayload = {
  submittedAt?: string
  projectName?: string
  finishLabel?: string
  selectedSolid?: string
  selectedFlake?: string
  compare?: number
  customer?: {
    customerName?: string
    phone?: string
    email?: string
    address?: string
    squareFeet?: string
    notes?: string
  }
  company?: {
    companyName?: string
    companyContact?: string
    brandColor?: string
  }
  assets?: {
    afterPreviewDataUrl?: string
  }
}

function json(res: any, status: number, body: unknown) {
  res.status(status).setHeader('Content-Type', 'application/json')
  res.send(JSON.stringify(body))
}

function makeLeadId() {
  const ts = new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14)
  const rand = Math.random().toString(36).slice(2, 6).toUpperCase()
  return `L-${ts}-${rand}`
}

function scoreLead(payload: QuotePayload) {
  const c = payload.customer ?? {}
  let score = 0

  if ((c.phone || '').trim()) score += 25
  if ((c.email || '').trim()) score += 20
  if ((c.address || '').trim()) score += 10

  const sqft = Number((c.squareFeet || '').replace(/[^0-9.]/g, ''))
  if (Number.isFinite(sqft)) {
    if (sqft >= 800) score += 25
    else if (sqft >= 500) score += 18
    else if (sqft >= 250) score += 10
  }

  const notes = (c.notes || '').toLowerCase()
  if (/(asap|urgent|this week|ready|book|schedule)/.test(notes)) score += 15
  if (/(budget|cheap|later|just browsing)/.test(notes)) score -= 8

  const clamped = Math.max(0, Math.min(100, score))
  const tier = clamped >= 70 ? 'hot' : clamped >= 40 ? 'warm' : 'cold'
  return { score: clamped, tier }
}

function getSlaHours(leadTier: string) {
  if (leadTier === 'hot') return 1
  if (leadTier === 'warm') return 8
  return 24
}

function computeNextFollowUpAt(leadTier: string, submittedAt?: string) {
  const base = submittedAt ? new Date(submittedAt) : new Date()
  const date = Number.isNaN(base.getTime()) ? new Date() : base
  date.setHours(date.getHours() + getSlaHours(leadTier))
  return date.toISOString()
}

function toText(payload: QuotePayload, leadId?: string, leadScore?: number, leadTier?: string, nextFollowUpAt?: string) {
  const c = payload.customer ?? {}
  return [
    `New Epoxy Quote Request`,
    `Lead ID: ${leadId ?? ''}`,
    `Submitted: ${payload.submittedAt ?? new Date().toISOString()}`,
    `Project: ${payload.projectName ?? ''}`,
    `Lead Score: ${leadScore ?? ''} (${leadTier ?? ''})`,
    `SLA Follow-up by: ${nextFollowUpAt ?? ''}`,
    `Finish: ${payload.finishLabel ?? ''}`,
    `Solid/Flake: ${payload.selectedSolid ?? ''} / ${payload.selectedFlake ?? ''}`,
    ``,
    `Customer: ${c.customerName ?? ''}`,
    `Phone: ${c.phone ?? ''}`,
    `Email: ${c.email ?? ''}`,
    `Address: ${c.address ?? ''}`,
    `SqFt: ${c.squareFeet ?? ''}`,
    `Notes: ${c.notes ?? ''}`,
  ].join('\n')
}

async function sendViaResend(payload: QuotePayload, leadId: string, leadScore: number, leadTier: string, nextFollowUpAt: string) {
  const apiKey = process.env.RESEND_API_KEY
  const to = process.env.RESEND_TO
  const from = process.env.RESEND_FROM
  if (!apiKey || !to || !from) return
  const recipients = to.split(/[;,]/).map((x) => x.trim()).filter(Boolean)
  if (!recipients.length) return

  const subject = `New quote request: ${payload.customer?.customerName || 'Unknown customer'} (${leadId}) [${leadTier.toUpperCase()} ${leadScore}]`

  const res = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from,
      to: recipients,
      subject,
      text: toText(payload, leadId, leadScore, leadTier, nextFollowUpAt),
    }),
  })

  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Resend owner email failed: ${res.status} ${body}`)
  }
}

async function forwardToWebhook(payload: QuotePayload, leadId: string, leadScore: number, leadTier: string, nextFollowUpAt: string) {
  const url = process.env.LEADS_FORWARD_WEBHOOK_URL
  if (!url) return

  const c = payload.customer ?? {}
  const forwarded = {
    leadId,
    leadScore,
    leadTier,
    status: 'new',
    owner: '',
    nextFollowUpAt,
    quotedPrice: '',
    outcome: '',
    submittedAt: payload.submittedAt ?? new Date().toISOString(),
    projectName: payload.projectName ?? '',
    finishLabel: payload.finishLabel ?? '',
    selectedSolid: payload.selectedSolid ?? '',
    selectedFlake: payload.selectedFlake ?? '',
    customerName: c.customerName ?? '',
    phone: c.phone ?? '',
    email: c.email ?? '',
    address: c.address ?? '',
    squareFeet: c.squareFeet ?? '',
    notes: c.notes ?? '',
    payload,
  }

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(forwarded),
  })

  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Webhook forward failed: ${res.status} ${body}`)
  }
}

async function sendCustomerAutoReply(payload: QuotePayload, leadId: string) {
  const apiKey = process.env.RESEND_API_KEY
  const from = process.env.RESEND_AUTOREPLY_FROM || process.env.RESEND_FROM
  const customerEmail = payload.customer?.email?.trim()
  if (!apiKey || !from || !customerEmail) return

  const companyName = payload.company?.companyName || 'Our Team'
  const customerName = payload.customer?.customerName || 'there'
  const subject = `We got your quote request (${leadId})`
  const text = [
    `Hi ${customerName},`,
    '',
    `Thanks for using our floor visualizer. We received your request and will follow up soon with your quote.`,
    '',
    `Lead ID: ${leadId}`,
    `Requested finish: ${payload.finishLabel ?? ''}`,
    '',
    `- ${companyName}`,
  ].join('\n')

  const res = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from,
      to: [customerEmail],
      subject,
      text,
    }),
  })

  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Resend auto-reply failed: ${res.status} ${body}`)
  }
}

export default async function handler(req: any, res: any) {
  if (req.method !== 'POST') return json(res, 405, { ok: false, error: 'Method not allowed' })

  const payload = req.body as QuotePayload
  const hasName = payload?.customer?.customerName?.trim()
  const hasContact = payload?.customer?.phone?.trim() || payload?.customer?.email?.trim()

  if (!hasName || !hasContact) {
    return json(res, 400, { ok: false, error: 'Missing required fields' })
  }

  const leadId = makeLeadId()
  const { score: leadScore, tier: leadTier } = scoreLead(payload)
  const nextFollowUpAt = computeNextFollowUpAt(leadTier, payload.submittedAt)
  const slaHours = getSlaHours(leadTier)

  try {
    const results = await Promise.allSettled([
      sendViaResend(payload, leadId, leadScore, leadTier, nextFollowUpAt),
      forwardToWebhook(payload, leadId, leadScore, leadTier, nextFollowUpAt),
      sendCustomerAutoReply(payload, leadId),
    ])

    const [ownerEmail, webhook, autoReply] = results
    const diagnostics = {
      ownerEmail: ownerEmail.status,
      webhook: webhook.status,
      autoReply: autoReply.status,
    }

    results.forEach((r, i) => {
      if (r.status === 'rejected') {
        const labels = ['ownerEmail', 'webhook', 'autoReply']
        console.error('quote pipeline error', labels[i], String(r.reason))
      }
    })

    return json(res, 200, { ok: true, leadId, leadScore, leadTier, slaHours, nextFollowUpAt, diagnostics })
  } catch {
    return json(res, 500, { ok: false, error: 'Quote handler failed' })
  }
}
