/**
 * Mirrors training/preprocess.py exactly: plain bilinear resize
 * (align_corners=false, no antialiasing, same as torch.nn.functional.interpolate)
 * so the shorter side is 256 px, centre crop 224x224, then ImageNet mean/std
 * normalisation into a CHW Float32Array.
 * test/parity.test.ts fails if the two drift apart.
 */
export const INPUT_SIZE = 224;
const RESIZE = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/** RGBA pixels, as from canvas ImageData or a decoded PNG. */
export interface Pixels {
  data: ArrayLike<number>;
  width: number;
  height: number;
}

/**
 * Source pixels and blend weight for output positions offset..offset+count of a
 * dst-sized resize (torch's area_pixel_compute_source_index, align_corners=false).
 */
function axis(srcSize: number, dstSize: number, offset: number, count: number) {
  const scale = srcSize / dstSize;
  const lo = new Int32Array(count);
  const hi = new Int32Array(count);
  const t = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const src = Math.max(scale * (offset + i + 0.5) - 0.5, 0);
    lo[i] = Math.floor(src);
    hi[i] = lo[i] < srcSize - 1 ? lo[i] + 1 : lo[i];
    t[i] = src - lo[i];
  }
  return { lo, hi, t };
}

export function preprocess({ data, width, height }: Pixels): Float32Array {
  const n = INPUT_SIZE;
  const scale = RESIZE / Math.min(width, height);
  const rh = Math.max(n, Math.round(height * scale));
  const rw = Math.max(n, Math.round(width * scale));
  const ys = axis(height, rh, Math.floor((rh - n) / 2), n);
  const xs = axis(width, rw, Math.floor((rw - n) / 2), n);

  const out = new Float32Array(3 * n * n);
  for (let y = 0; y < n; y++) {
    const r0 = ys.lo[y] * width;
    const r1 = ys.hi[y] * width;
    const ty = ys.t[y];
    for (let x = 0; x < n; x++) {
      const tx = xs.t[x];
      const a = (r0 + xs.lo[x]) * 4;
      const b = (r0 + xs.hi[x]) * 4;
      const c = (r1 + xs.lo[x]) * 4;
      const d = (r1 + xs.hi[x]) * 4;
      for (let ch = 0; ch < 3; ch++) {
        const top = data[a + ch] * (1 - tx) + data[b + ch] * tx;
        const bottom = data[c + ch] * (1 - tx) + data[d + ch] * tx;
        const v = (top * (1 - ty) + bottom * ty) / 255;
        out[ch * n * n + y * n + x] = (v - MEAN[ch]) / STD[ch];
      }
    }
  }
  return out;
}
