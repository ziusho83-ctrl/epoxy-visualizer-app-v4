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

const FLOOR_PROMPT = 'garage floor,concrete floor,epoxy floor,ground,pavement'
const NEGATIVE_PROMPT = 'wall,ceiling,garage door,cabinet,door,shelf,car,sky,roof,driveway,window,light fixture,beam,rafter'

// Fallback prompts for when primary detection fails (e.g. dark stained floors)
const FALLBACK_FLOOR_PROMPT = 'concrete surface,ground surface,flat ground,floor surface,slab'
const FALLBACK_NEGATIVE_PROMPT = 'wall,ceiling,sky,roof,door,window,car'

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
  const sumYs: number[] = [0] // track sum of Y values for centroid calculation
  const stack: number[] = []
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (occ[i] === 1 && labels[i] === 0) {
        const label = sizes.length
        sizes.push(0)
        botYs.push(0)
        sumYs.push(0)
        stack.push(i)
        labels[i] = label
        let count = 0
        let maxY = 0
        let sumY = 0
        while (stack.length > 0) {
          const k = stack.pop()!
          const ky = Math.floor(k / w)
          const kx = k - ky * w
          count++
          if (ky > maxY) maxY = ky
          sumY += ky
          if (kx > 0 && occ[k - 1] === 1 && labels[k - 1] === 0) { labels[k - 1] = label; stack.push(k - 1) }
          if (kx < w - 1 && occ[k + 1] === 1 && labels[k + 1] === 0) { labels[k + 1] = label; stack.push(k + 1) }
          if (ky > 0 && occ[k - w] === 1 && labels[k - w] === 0) { labels[k - w] = label; stack.push(k - w) }
          if (ky < h - 1 && occ[k + w] === 1 && labels[k + w] === 0) { labels[k + w] = label; stack.push(k + w) }
        }
        sizes[label] = count
        botYs[label] = maxY
        sumYs[label] = sumY
      }
    }
  }

  // Filter by min area + centroid position, then pick floor component.
  // Reject components whose centroid is in the top 40% of the image (ceiling/wall).
  // Among remaining, prefer lowest botY, break ties by size.
  const CEILING_CUTOFF = 0.30 // centroid must be below this fraction of image height
  let bestLabel = 0
  let bestBotY = -1
  let bestSize = 0
  let candidateCount = 0
  for (let lbl = 1; lbl < sizes.length; lbl++) {
    if (sizes[lbl] < totalPixels * MIN_AREA_FRAC) continue // skip tiny strips
    const centroidY = sumYs[lbl] / sizes[lbl]
    const centroidFrac = centroidY / h
    console.log(`  component ${lbl}: size=${sizes[lbl]} (${(sizes[lbl]/totalPixels*100).toFixed(1)}%) centroidY=${centroidFrac.toFixed(2)} botY=${botYs[lbl]}`)
    if (centroidFrac < CEILING_CUTOFF) {
      console.log(`  -> rejected (centroid in top ${(CEILING_CUTOFF*100).toFixed(0)}%, likely ceiling/wall)`)
      continue
    }
    candidateCount++
    if (
      botYs[lbl] > bestBotY ||
      (botYs[lbl] === bestBotY && sizes[lbl] > bestSize)
    ) {
      bestLabel = lbl
      bestBotY = botYs[lbl]
      bestSize = sizes[lbl]
    }
  }
  if (bestLabel === 0) {
    console.log(`No valid floor component found (${sizes.length - 1} total, ${candidateCount} passed filters)`)
    return []
  }
  console.log(`Grounded SAM mask: ${sizes.length - 1} components, picked label ${bestLabel} size=${bestSize} botY=${bestBotY} centroidY=${(sumYs[bestLabel]/sizes[bestLabel]/h).toFixed(2)} (0.5% min, ${(CEILING_CUTOFF*100).toFixed(0)}% ceiling cutoff)`)

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

    // --- Helper: run Grounded SAM with given prompts ---
    async function runSegmentation(
      floorPrompt: string,
      negPrompt: string,
      label: string
    ): Promise<Pt[] | null> {
      console.log(`[${label}] Running with prompt: "${floorPrompt}"`)
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
              mask_prompt: floorPrompt,
              negative_mask_prompt: negPrompt,
              adjustment_factor: 0,
            },
          }),
        })

        if (createRes.status === 429) {
          console.log(`[${label}] Replicate 429 on attempt ${attempt + 1}, retrying...`)
          await new Promise((r) => setTimeout(r, (attempt + 1) * 3000))
          continue
        }

        if (!createRes.ok) {
          const err = await createRes.text()
          console.error(`[${label}] Replicate create error:`, createRes.status, err)
          return null
        }

        result = await createRes.json()
        break
      }

      if (!result) return null

      // Poll until complete
      const maxWait = 160_000
      const start = Date.now()
      while (result.status !== 'succeeded' && result.status !== 'failed' && result.status !== 'canceled') {
        if (Date.now() - start > maxWait) {
          console.error(`[${label}] Timed out`)
          return null
        }
        await new Promise((r) => setTimeout(r, 1500))
        const pollRes = await fetch(result.urls.get, {
          headers: { Authorization: `Bearer ${REPLICATE_API_TOKEN}` },
        })
        result = await pollRes.json()
      }

      if (result.status === 'failed') {
        console.error(`[${label}] Replicate failed:`, result.error)
        return null
      }

      // Download mask
      const output = result.output
      if (!Array.isArray(output) || output.length < 3) {
        console.error(`[${label}] Unexpected output shape:`, output)
        return null
      }
      const maskUrl = output[2]
      const maskRes = await fetch(maskUrl)
      if (!maskRes.ok) {
        console.error(`[${label}] Failed to download mask`)
        return null
      }
      const maskBuf = Buffer.from(await maskRes.arrayBuffer())

      const rawContour = await traceMaskContour(maskBuf)
      if (rawContour.length < 3) {
        console.log(`[${label}] No valid contour found`)
        return null
      }

      console.log(`[${label}] Got ${rawContour.length} raw contour points`)
      return rawContour
    }

    // --- Try primary prompt, then fallback if no floor detected ---
    let rawContour = await runSegmentation(FLOOR_PROMPT, NEGATIVE_PROMPT, 'primary')

    if (!rawContour) {
      console.log('Primary detection failed, trying fallback prompt...')
      rawContour = await runSegmentation(FALLBACK_FLOOR_PROMPT, FALLBACK_NEGATIVE_PROMPT, 'fallback')
    }

    if (!rawContour) {
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
