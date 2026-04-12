# Lead Pipeline Status

## Already completed
- Client quote form (#5) submits to `/api/quote`.
- API generates Lead ID.
- API can send owner email (Resend), forward webhook payload, and customer auto-reply.
- Production deployed: https://epoxy-visualizer-app.vercel.app

## Blocked by missing credentials/targets
No Vercel env vars are configured yet.

Required to activate owner email alerts:
- `RESEND_API_KEY`
- `RESEND_FROM`
- `RESEND_TO`

Optional for customer confirmation email:
- `RESEND_AUTOREPLY_FROM`

Required to activate Google Sheet/automation logging:
- `LEADS_FORWARD_WEBHOOK_URL`

## Payload fields forwarded to webhook
- leadId
- submittedAt
- projectName
- finishLabel
- selectedSolid
- selectedFlake
- customerName
- phone
- email
- address
- squareFeet
- notes
- payload (raw full payload)

## Next action for Vu
1) Set Vercel env vars above.
2) (Optional) Use `GOOGLE_SHEETS_WEBHOOK_TEMPLATE.gs` to create a Sheet receiver URL and set it as `LEADS_FORWARD_WEBHOOK_URL`.
3) Redeploy (or trigger a new deployment).
4) Submit a test lead from the app.
