// Цветовая шкала уверенности модели талька — "холодный → горячий"
// (синий = едва прошёл порог, красный = максимальная уверенность).
// Используется и для отрисовки слоя (ImageViewer), и для легенды (TalcMaskEditor) —
// единственный источник цветов, чтобы они не разъезжались.
//
// Чёрный и зелёный намеренно исключены: чёрный — цвет фона шлифа (тень слилась
// бы с породой), зелёный — уже занят под "рядовые срастания" на слое "Тип".
export const CONFIDENCE_COLOR_STOPS: [number, number, number, number][] = [
  [0, 50, 90, 220],
  [0.33, 170, 60, 200],
  [0.66, 240, 140, 40],
  [1, 220, 40, 40],
];

// Значение в PNG с картой уверенности (0..255) внутри маски никогда не
// опускается ниже этого байта — порог модели prob>0.5 (0.5*255=127.5).
// Растягиваем именно этот диапазон на всю шкалу, иначе "холодная" половина
// цветов никогда бы не показывалась.
export const CONFIDENCE_BYTE_FLOOR = 128;

export function confidenceByteToT(byte: number): number {
  return Math.min(1, Math.max(0, (byte - CONFIDENCE_BYTE_FLOOR) / (255 - CONFIDENCE_BYTE_FLOOR)));
}

export function confidenceColor(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  for (let i = 0; i < CONFIDENCE_COLOR_STOPS.length - 1; i++) {
    const [t0, r0, g0, b0] = CONFIDENCE_COLOR_STOPS[i];
    const [t1, r1, g1, b1] = CONFIDENCE_COLOR_STOPS[i + 1];
    if (clamped <= t1 || i === CONFIDENCE_COLOR_STOPS.length - 2) {
      const f = Math.min(1, Math.max(0, (clamped - t0) / (t1 - t0 || 1)));
      return [
        Math.round(r0 + (r1 - r0) * f),
        Math.round(g0 + (g1 - g0) * f),
        Math.round(b0 + (b1 - b0) * f),
      ];
    }
  }
  const [, r, g, b] = CONFIDENCE_COLOR_STOPS[CONFIDENCE_COLOR_STOPS.length - 1];
  return [r, g, b];
}

export const CONFIDENCE_CSS_GRADIENT = `linear-gradient(to right, ${CONFIDENCE_COLOR_STOPS.map(
  ([t, r, g, b]) => `rgb(${r}, ${g}, ${b}) ${t * 100}%`
).join(", ")})`;

// Постоянная альфа overlay-слоя уверенности — та же плотность, что и у
// обычной маски (0.55), чтобы режимы не отличались по "весу" картинки, а
// только по цвету.
export const CONFIDENCE_DISPLAY_ALPHA = Math.round(255 * 0.55);
