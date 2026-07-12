import '@testing-library/jest-dom/vitest'

Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  value: () => ({
    createImageData: (width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) }),
    putImageData: () => undefined,
  }),
})
