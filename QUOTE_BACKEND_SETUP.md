# Quote Backend Setup (GoDaddy site + Visualizer app)

## What is already wired
- Frontend submits quote requests to `POST /api/quote` by default.
- API route: `api/quote.ts`
- Payload includes customer info + selected finish + generated after-preview image (data URL).

## Fastest production setup (Vercel)
1. Push this app repo to GitHub.
2. Import project in Vercel.
3. Add env vars in Vercel Project Settings:
   - `RESEND_API_KEY`
   - `RESEND_FROM` (verified sender, e.g. `quotes@yourdomain.com`)
   - `RESEND_TO` (your inbox)
   - Optional: `LEADS_FORWARD_WEBHOOK_URL` (Zapier/Make/Google Apps Script)
4. Redeploy.

Now every submitted quote will hit `/api/quote`, generate a Lead ID, and email you.
If customer email is provided, it can also send an auto-reply confirmation.

## Optional: Send leads to Google Sheet too
Use a no-code automation:
- Zapier/Make webhook trigger receives JSON
- Append row to Google Sheet
- Optional: send SMS/email notification

Set the webhook URL as:
- `LEADS_FORWARD_WEBHOOK_URL=https://hooks.zapier.com/...`

## GoDaddy website attach
- Add button/menu item: **Try Garage Visualizer**
- Link to deployed app URL (or subdomain like `visualizer.yourdomain.com`)

## Recommended quote SLA text on website
"Get your personalized estimate within 1 business day."
