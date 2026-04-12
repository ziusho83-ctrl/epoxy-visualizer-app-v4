/**
 * GET /api/warmup (V3)
 *
 * Fires a lightweight Grounded SAM prediction to keep the model warm on Replicate.
 * Called on page load + every 60s to reduce cold start delays for Auto Detect.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node'

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN
const MODEL_VERSION = 'ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c'

// Tiny 1x1 pixel PNG to trigger model warmup
const TINY_IMAGE = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=='

export default async function handler(_req: VercelRequest, res: VercelResponse) {
  if (!REPLICATE_API_TOKEN) {
    return res.status(200).json({ status: 'no_token' })
  }

  try {
    await fetch('https://api.replicate.com/v1/predictions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${REPLICATE_API_TOKEN}`,
      },
      body: JSON.stringify({
        version: MODEL_VERSION,
        input: {
          image: TINY_IMAGE,
          mask_prompt: 'floor',
          negative_mask_prompt: 'wall',
        },
      }),
    })

    return res.status(200).json({ status: 'warming' })
  } catch {
    return res.status(200).json({ status: 'error' })
  }
}
