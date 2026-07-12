export function clampFrame(frame: number, lastFrame: number): number {
  return Math.min(Math.max(Math.round(frame), 0), Math.max(lastFrame, 0))
}

export function resetFrame(): number {
  return 0
}

export function advanceFrame(frame: number, elapsedMs: number, lastFrame: number, speed: number, frameDurationMs: number): number {
  const steps = Math.floor((elapsedMs * speed) / frameDurationMs)
  return clampFrame(frame + steps, lastFrame)
}
