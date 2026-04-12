// Google Apps Script: quote lead webhook receiver
// 1) Create a Google Sheet and set tab name to: Leads
// 2) Extensions -> Apps Script, paste this file, save
// 3) Deploy -> New deployment -> Web app
//    - Execute as: Me
//    - Who has access: Anyone
// 4) Copy the Web app URL and set as LEADS_FORWARD_WEBHOOK_URL in Vercel

// Optional stale-lead reminder config
const ALERT_EMAILS = ['ziusho83@gmail.com', 'hoquocvu28@gmail.com']

function doPost(e) {
  try {
    const payload = JSON.parse(e.postData.contents || '{}')

    const ss = SpreadsheetApp.getActiveSpreadsheet()
    const sheet = ss.getSheetByName('Leads') || ss.insertSheet('Leads')

    if (sheet.getLastRow() === 0) {
      sheet.appendRow([
        'leadId',
        'leadScore',
        'leadTier',
        'status',
        'slaHours',
        'owner',
        'nextFollowUpAt',
        'quotedPrice',
        'outcome',
        'submittedAt',
        'customerName',
        'phone',
        'email',
        'address',
        'squareFeet',
        'projectName',
        'finishLabel',
        'selectedSolid',
        'selectedFlake',
        'notes',
      ])
    }

    sheet.appendRow([
      payload.leadId || '',
      payload.leadScore || '',
      payload.leadTier || '',
      payload.status || 'new',
      payload.slaHours || '',
      payload.owner || '',
      payload.nextFollowUpAt || '',
      payload.quotedPrice || '',
      payload.outcome || '',
      payload.submittedAt || '',
      payload.customerName || '',
      payload.phone || '',
      payload.email || '',
      payload.address || '',
      payload.squareFeet || '',
      payload.projectName || '',
      payload.finishLabel || '',
      payload.selectedSolid || '',
      payload.selectedFlake || '',
      payload.notes || '',
    ])

    return ContentService
      .createTextOutput(JSON.stringify({ ok: true }))
      .setMimeType(ContentService.MimeType.JSON)
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ ok: false, error: String(err) }))
      .setMimeType(ContentService.MimeType.JSON)
  }
}

// Run every hour via time-driven trigger.
function checkStaleLeads() {
  const ss = SpreadsheetApp.getActiveSpreadsheet()
  const sheet = ss.getSheetByName('Leads')
  if (!sheet) return

  const values = sheet.getDataRange().getValues()
  if (values.length < 2) return

  const headers = values[0].map(String)
  const idx = Object.fromEntries(headers.map((h, i) => [h, i]))
  const now = new Date()
  const stale = []

  for (let r = 1; r < values.length; r++) {
    const row = values[r]
    const status = String(row[idx.status] || '').toLowerCase()
    if (['won', 'lost', 'archived'].includes(status)) continue

    const dueRaw = row[idx.nextFollowUpAt]
    if (!dueRaw) continue
    const due = new Date(dueRaw)
    if (Number.isNaN(due.getTime())) continue
    if (due <= now) {
      stale.push({
        leadId: row[idx.leadId] || '',
        tier: row[idx.leadTier] || '',
        score: row[idx.leadScore] || '',
        customer: row[idx.customerName] || '',
        phone: row[idx.phone] || '',
        status: row[idx.status] || '',
        due: due.toISOString(),
      })
    }
  }

  if (!stale.length) return

  const lines = stale.slice(0, 25).map((x) =>
    `${x.leadId} | ${x.tier}/${x.score} | ${x.customer} | ${x.phone} | ${x.status} | due ${x.due}`,
  )

  const subject = `[Leads] ${stale.length} stale follow-up${stale.length > 1 ? 's' : ''}`
  const body = `These leads are overdue for follow-up:\n\n${lines.join('\n')}`
  MailApp.sendEmail(ALERT_EMAILS.join(','), subject, body)
}

function setupHourlyStaleLeadTrigger() {
  const handlers = ScriptApp.getProjectTriggers().filter((t) => t.getHandlerFunction() === 'checkStaleLeads')
  handlers.forEach((t) => ScriptApp.deleteTrigger(t))
  ScriptApp.newTrigger('checkStaleLeads').timeBased().everyHours(1).create()
}
