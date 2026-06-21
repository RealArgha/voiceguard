export const colors = {
  bg:        '#080c14',
  surface:   '#0f1623',
  surface2:  '#151d2e',
  border:    'rgba(255,255,255,0.07)',
  text:      '#e2e8f0',
  muted:     '#64748b',
  accent:    '#6366f1',

  low:       '#22c55e',
  lowGlow:   'rgba(34,197,94,0.2)',
  medium:    '#f59e0b',
  mediumGlow:'rgba(245,158,11,0.2)',
  high:      '#ef4444',
  highGlow:  'rgba(239,68,68,0.2)',
} as const;

export type Band = 'LOW' | 'MEDIUM' | 'HIGH';

export function bandColor(band: Band) {
  if (band === 'HIGH')   return colors.high;
  if (band === 'MEDIUM') return colors.medium;
  return colors.low;
}

export function bandGlow(band: Band) {
  if (band === 'HIGH')   return colors.highGlow;
  if (band === 'MEDIUM') return colors.mediumGlow;
  return colors.lowGlow;
}
