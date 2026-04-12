/**
 * POST /api/auto-mask (V4)
 *
 * Uses Grounded SAM (Grounding DINO + SAM) via Replicate to segment the garage
 * floor using text prompts. This replaces Mask2Former's scene-parsing approach
 * (which hallucinated ceilings as floors) with a text-prompted segmentation
 * that asks explicitly for "garage floor" and excludes walls/ceiling/doors.
 *
 * Body: { imageDataUrl: string }
 * Returns: { points: { x: number, y: number }[] }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node'
import sharp from 'sharp'
import { contours } from 'd3-contour'

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN
// schananas/grounded_sam: Grounding DINO + SAM with text prompts
const MODEL_VERSION = 'ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c'

const FLOOR_PROMPT = 'garage floor,concrete floor,epoxy floor'
const NEGATIVE_PROMPT = 'wall,ceiling,garage door,cabinet,door,shelf,car'

type Pt = { x: number; y: number }

/**
 * Trace the outer contour of a binary mask using Moore-neighbor boundary
 * following. Returns a list of (x, y) pixel points along the boundary of
 * the largest connected component. Then converts to normalized 0-100 space.
 */
async function traceMaskContour(maskBuf: Buffer): Promise<Pt[]> {
  // Decode the mask (PNG or JPEG) to raw grayscale using sharp
  const { data, info } = await sharp(maskBuf)
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true })
  const w = info.width
  const h = info.height

  // Build a binary occupancy grid. Mask is white=foreground, black=background.
  // Also handle the inverse case just in case.
  const occ = new Uint8Array(w * h)
  let whiteCount = 0
  for (let i = 0; i < w * h; i++) {
    if (data[i] > 128) {
      occ[i] = 1
      whiteCount++
    }
  }
  // If white is the majority, the mask is probably inverted
  if (whiteCount > w * h * 0.5) {
    for (let i = 0; i < w * h; i++) occ[i] = occ[i] === 1 ? 0 : 1
  }

  // --- Find connected components (4-connectivity flood fill) ---
  // REVERSAL POINT: This uses "prefer component whose botY reaches lowest,
  // break ties by size" with 0.5% min threshold. To switch back to pure
  // largest-component selection with 5% threshold, change MIN_AREA_FRAC to
  // 0.05 and remove the botY preference (just pick by bestSize).
  const MIN_AREA_FRAC = 0.005 // 0.5% of image — filters driveway strips
  const totalPixels = w * h
  const labels = new Int32Array(w * h)
  const sizes: number[] = [0]
  const botYs: number[] = [0] // track max Y (bottom-most row) per component
  const stack: number[] = []
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (occ[i] === 1 && labels[i] === 0) {
        const label = sizes.length
        sizes.push(0)
        botYs.push(0)
        stack.push(i)
        labels[i] = label
        let count = 0
        let maxY = 0
        while (stack.length > 0) {
          const k = stack.pop()!
          const ky = Math.floor(k / w)
          const kx = k - ky * w
          count++
          if (ky > maxY) maxY = ky
          if (kx > 0 && occ[k - 1] === 1 && labels[k - 1] === 0) { labels[k - 1] = label; stack.push(k - 1) }
          if (kx < w - 1 && occ[k + 1] === 1 && labels[k + 1] === 0) { labels[k + 1] = label; stack.push(k + 1) }
          if (ky > 0 && occ[k - w] === 1 && labels[k - w] === 0) { labels[k - w] = label; stack.push(k - w) }
          if (ky < h - 1 && occ[k + w] === 1 && labels[k + w] === 0) { labels[k + w] = label; stack.push(k + w) }
        }
        sizes[label] = count
        botYs[label] = maxY
      }
    }
  }

  // Filter by min area, then pick the component whose bottom edge reaches
  // lowest (highest Y value). Break ties by size (largest wins).
  let bestLabel = 0
  let bestBotY = -1
  let bestSize = 0
  for (let lbl = 1; lbl < sizes.length; lbl++) {
    if (sizes[lbl] < totalPixels * MIN_AREA_FRAC) continue // skip tiny strips
    if (
      botYs[lbl] > bestBotY ||
      (botYs[lbl] === bestBotY && sizes[lbl] > bestSize)
    ) {
      bestLabel = lbl
      bestBotY = botYs[lbl]
      bestSize = sizes[lbl]
    }
  }
  if (bestLabel === 0) return []
  console.log(`Grounded SAM mask: ${sizes.length - 1} components, picked label ${bestLabel} size=${bestSize} botY=${bestBotY} (0.5% min threshold)`)

  // Keep only the selected component
  const blob = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) {
    if (labels[i] === bestLabel) blob[i] = 1
  }

  // --- Gaussian blur + Marching Squares contour extraction ---
  // Create a buffer from only the selected component, smooth it, then extract contour
  const blobBuf = Buffer.alloc(w * h)
  for (let i = 0; i < w * h; i++) {
    if (blob[i] === 1) blobBuf[i] = 255
  }

  // Apply Gaussian blur (sigma ~3) then re-threshold to smooth jagged edges
  const smoothedBuf = await sharp(blobBuf, { raw: { width: w, height: h, channels: 1 } })
    .blur(3)
    .threshold(127)
    .toColourspace("b-w")
    .raw()
    .toBuffer()

  // Build Float64Array for d3-contour (1 = foreground, 0 = background)
  const values = new Float64Array(w * h)
  for (let i = 0; i < w * h; i++) {
    values[i] = smoothedBuf[i] > 127 ? 1 : 0
  }

  // Extract 0.5 isoline using marching squares
  const contourGenerator = contours().size([w, h])
  const isoContours = contourGenerator.contour(Array.from(values), 0.5)

  // isoContours.coordinates is MultiPolygon: number[][][][]
  // Pick the longest ring (outer boundary)
  let longest: Array<[number, number]> = []
  for (const polygon of isoContours.coordinates) {
    for (const ring of polygon) {
      if (ring.length > longest.length) {
        longest = ring as Array<[number, number]>
      }
    }
  }

  if (longest.length < 3) return []

  console.log(`Marching squares contour: ${longest.length} raw points`)

  // Convert to {x, y} normalized 0-100
  const rawContour = longest.map(([px, py]) => ({
    x: (px / w) * 100,
    y: (py / h) * 100,
  }))

  return rawContour
}

// Legacy: kept for reference, replaced by visvalingamWhyatt above
// function douglasPeucker(pts: Pt[], epsilon: number): Pt[] {
//   if (pts.length < 3) return pts.slice()
//
//   const perpDist = (p: Pt, a: Pt, b: Pt): number => {
//     const dx = b.x - a.x
//     const dy = b.y - a.y
//     const mag = Math.sqrt(dx * dx + dy * dy)
//     if (mag < 1e-9) {
//       const dxp = p.x - a.x
//       const dyp = p.y - a.y
//       return Math.sqrt(dxp * dxp + dyp * dyp)
//     }
//     return Math.abs(dy * p.x - dx * p.y + b.x * a.y - b.y * a.x) / mag
//   }
//
//   const simplify = (start: number, end: number): Pt[] => {
//     if (end - start < 2) return [pts[start]]
//     let maxDist = 0
//     let maxIdx = start
//     for (let i = start + 1; i < end; i++) {
//       const d = perpDist(pts[i], pts[start], pts[end])
//       if (d > maxDist) {
//         maxDist = d
//         maxIdx = i
//       }
//     }
//     if (maxDist > epsilon) {
//       const left = simplify(start, maxIdx)
//       const right = simplify(maxIdx, end)
//       return left.concat(right.slice(1))
//     }
//     return [pts[start], pts[end]]
//   }
//
//   return simplify(0, pts.length - 1)
// }

/**
 * Visvalingam-Whyatt simplification: iteratively removes the point forming
 * the smallest triangle area with its neighbors until targetCount is reached.
 * Better than Douglas-Peucker for preserving shape character.
 */
function visvalingamWhyatt(pts: Pt[], targetCount: number): Pt[] {
  if (pts.length <= targetCount) return pts.slice();

  type VWNode = {
    point: Pt;
    area: number;
    prev: VWNode | null;
    next: VWNode | null;
    removed: boolean;
  };

  const nodes: VWNode[] = pts.map(p => ({
    point: p, area: Infinity, prev: null, next: null, removed: false
  }));
  for (let i = 0; i < nodes.length; i++) {
    nodes[i].prev = i > 0 ? nodes[i - 1] : null;
    nodes[i].next = i < nodes.length - 1 ? nodes[i + 1] : null;
  }

  function triangleArea(a: Pt, b: Pt, c: Pt): number {
    return Math.abs((a.x - c.x) * (b.y - a.y) - (a.x - b.x) * (c.y - a.y)) / 2;
  }

  function calcArea(node: VWNode): number {
    if (!node.prev || !node.next) return Infinity;
    return triangleArea(node.prev.point, node.point, node.next.point);
  }

  for (const n of nodes) n.area = calcArea(n);

  let remaining = pts.length;
  while (remaining > targetCount) {
    let minNode: VWNode | null = null;
    let minArea = Infinity;
    let current: VWNode | null = nodes[0].next;
    while (current && current.next) {
      if (current.area < minArea) {
        minArea = current.area;
        minNode = current;
      }
      current = current.next;
    }
    if (!minNode) break;

    // Remove it
    if (minNode.prev) minNode.prev.next = minNode.next;
    if (minNode.next) minNode.next.prev = minNode.prev;
    minNode.removed = true;

    // Recalculate neighbors with monotonicity
    if (minNode.prev) minNode.prev.area = Math.max(calcArea(minNode.prev), minArea);
    if (minNode.next) minNode.next.area = Math.max(calcArea(minNode.next), minArea);

    remaining--;
  }

  const result: Pt[] = [];
  let node: VWNode | null = nodes[0];
  while (node) {
    result.push(node.point);
    node = node.next;
  }
  return result;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'POST only' })
  }

  if (!REPLICATE_API_TOKEN) {
    return res.status(500).json({ error: 'REPLICATE_API_TOKEN not configured' })
  }

  try {
    const { imageDataUrl } = req.body as { imageDataUrl?: string }

    if (!imageDataUrl || !imageDataUrl.startsWith('data:image/')) {
      return res.status(400).json({ error: 'Missing or invalid imageDataUrl' })
    }

    if (imageDataUrl.length > 14_000_000) {
      return res.status(413).json({ error: 'Image too large (max ~10 MB)' })
    }

    // --- Step 1: Create Grounded SAM prediction (with retry for 429) ---
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let result: any = null
    for (let attempt = 0; attempt < 3; attempt++) {
      const createRes = await fetch('https://api.replicate.com/v1/predictions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${REPLICATE_API_TOKEN}`,
        },
        body: JSON.stringify({
          version: MODEL_VERSION,
          input: {
            image: imageDataUrl,
            mask_prompt: FLOOR_PROMPT,
            negative_mask_prompt: NEGATIVE_PROMPT,
            adjustment_factor: 0,
          },
        }),
      })

      if (createRes.status === 429) {
        console.log(`Replicate 429 on attempt ${attempt + 1}, retrying...`)
        await new Promise((r) => setTimeout(r, (attempt + 1) * 3000))
        continue
      }

      if (!createRes.ok) {
        const err = await createRes.text()
        console.error('Replicate create error:', createRes.status, err)
        return res.status(502).json({ error: `Replicate API error (${createRes.status})` })
      }

      result = await createRes.json()
      break
    }

    if (!result) {
      return res.status(429).json({ error: 'Rate limited by Replicate, try again in a moment' })
    }

    // --- Step 2: Poll until complete ---
    const maxWait = 160_000
    const start = Date.now()
    while (result.status !== 'succeeded' && result.status !== 'failed' && result.status !== 'canceled') {
      if (Date.now() - start > maxWait) {
        return res.status(504).json({ error: 'Segmentation timed out (model may be cold-starting, try again)' })
      }
      await new Promise((r) => setTimeout(r, 1500))
      const pollRes = await fetch(result.urls.get, {
        headers: { Authorization: `Bearer ${REPLICATE_API_TOKEN}` },
      })
      result = await pollRes.json()
    }

    if (result.status === 'failed') {
      console.error('Replicate failed:', result.error)
      return res.status(502).json({ error: 'Segmentation failed', detail: result.error })
    }

    // --- Step 3: Download mask and trace contour ---
    // grounded_sam output is an array: [annotated, neg_annotated, mask, inverted_mask]
    // We want the positive mask (index 2)
    const output = result.output
    if (!Array.isArray(output) || output.length < 3) {
      console.error('Unexpected Grounded SAM output shape:', output)
      return res.status(502).json({ error: 'No segmentation mask returned' })
    }
    const maskUrl = output[2]
    const maskRes = await fetch(maskUrl)
    if (!maskRes.ok) {
      return res.status(502).json({ error: 'Failed to download segmentation mask' })
    }
    const maskBuf = Buffer.from(await maskRes.arrayBuffer())

    // Trace the largest connected mask contour
    const rawContour = await traceMaskContour(maskBuf)
    if (rawContour.length < 3) {
      return res.status(200).json({ points: [], error: 'No floor detected in this image' })
    }

    // Visvalingam-Whyatt: target 120 points (preserves corners)
    const MAX_POINTS = 120
    const final = visvalingamWhyatt(rawContour, MAX_POINTS)
    console.log(`VW simplified: ${rawContour.length} → ${final.length}`)

    return res.status(200).json({ points: final })
  } catch (err) {
    console.error('auto-mask handler error:', err)
    return res.status(500).json({ error: 'Internal server error' })
  }
}
