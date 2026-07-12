import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { clampFrame } from './playback'

export function useAnimationPlayback(frameCount: number, frameDurationMs = 180) {
  const [frame, setFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [reducedMotion, setReducedMotion] = useState(false)
  const requestRef = useRef<number | null>(null)
  const previousRef = useRef<number | null>(null)
  const remainderRef = useRef(0)
  const lastFrame = Math.max(0, frameCount - 1)

  useEffect(() => {
    const query = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!query) return
    const update = () => setReducedMotion(query.matches)
    update(); query.addEventListener('change', update)
    return () => query.removeEventListener('change', update)
  }, [])
  useEffect(() => { setFrame((current) => clampFrame(current, lastFrame)); setIsPlaying(false); previousRef.current = null }, [lastFrame])
  useEffect(() => { if (reducedMotion) setIsPlaying(false) }, [reducedMotion])
  useEffect(() => {
    if (!isPlaying || reducedMotion || lastFrame === 0) return
    const tick = (now: number) => {
      const previous = previousRef.current ?? now
      previousRef.current = now
      remainderRef.current += (now - previous) * speed
      const steps = Math.floor(remainderRef.current / frameDurationMs)
      if (steps > 0) {
        remainderRef.current -= steps * frameDurationMs
        setFrame((current) => {
          const next = clampFrame(current + steps, lastFrame)
          if (next === lastFrame) setIsPlaying(false)
          return next
        })
      }
      requestRef.current = requestAnimationFrame(tick)
    }
    requestRef.current = requestAnimationFrame(tick)
    return () => { if (requestRef.current !== null) cancelAnimationFrame(requestRef.current); requestRef.current = null; previousRef.current = null }
  }, [frameDurationMs, isPlaying, lastFrame, reducedMotion, speed])
  const reset = useCallback(() => { setIsPlaying(false); remainderRef.current = 0; setFrame(0) }, [])
  const step = useCallback((amount: number) => { setIsPlaying(false); setFrame((current) => clampFrame(current + amount, lastFrame)) }, [lastFrame])
  return useMemo(() => ({ frame, isPlaying, speed, reducedMotion, canPlay: lastFrame > 0 && !reducedMotion, play: () => setIsPlaying(true), pause: () => setIsPlaying(false), reset, step, setFrame: (next: number) => { setIsPlaying(false); setFrame(clampFrame(next, lastFrame)) }, setSpeed }), [frame, isPlaying, speed, reducedMotion, lastFrame, reset, step])
}
