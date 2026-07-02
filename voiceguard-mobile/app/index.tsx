import React from 'react';
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, SafeAreaView,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import RingGauge from '../components/RingGauge';
import { useVoiceStream } from '../hooks/useVoiceStream';
import { colors, bandColor } from '../constants/theme';

function statusLabel(status: string, recording: boolean): string {
  if (recording)           return '● LIVE';
  if (status === 'no_mic') return 'NO MIC';
  return status.toUpperCase();
}

export default function MonitorScreen() {
  const { status, event, start, stop } = useVoiceStream();

  const recording  = status === 'recording';
  const noMic      = status === 'no_mic';
  const band       = event?.band   ?? 'LOW';
  const score      = event?.score  ?? 0;
  const action     = event?.action ?? '—';
  const cnn        = event ? (event.cnn_prob * 100).toFixed(1) + '%' : '—';
  const keywords   = event?.keywords ?? [];
  const explanation = noMic
    ? 'No microphone detected. Run on a physical device to analyse live calls.'
    : (event?.explanation || 'Tap the mic button to start monitoring.');

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />

      {/* header */}
      <View style={styles.header}>
        <View style={styles.logoIcon}>
          <Text style={styles.logoEmoji}>🛡️</Text>
        </View>
        <Text style={styles.logoText}>VoiceGuard</Text>
        <View style={[
          styles.statusPill,
          recording && styles.statusPillLive,
          noMic     && styles.statusPillMuted,
        ]}>
          <Text style={[
            styles.statusText,
            recording && styles.statusTextLive,
          ]}>
            {statusLabel(status, recording)}
          </Text>
        </View>
      </View>

      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        {/* ring gauge */}
        <View style={styles.gaugeWrap}>
          <RingGauge score={score} band={band} />
        </View>

        {/* mic button */}
        <TouchableOpacity
          style={[
            styles.micBtn,
            recording && { backgroundColor: colors.high },
            noMic     && styles.micBtnDisabled,
          ]}
          onPress={recording ? stop : start}
          disabled={noMic}
          activeOpacity={0.8}
        >
          <Text style={styles.micEmoji}>
            {noMic ? '🚫' : recording ? '⏹' : '🎙️'}
          </Text>
        </TouchableOpacity>

        {noMic && (
          <Text style={styles.noMicHint}>
            Physical device required for mic input
          </Text>
        )}

        {/* action card */}
        <View style={[styles.card, styles.actionCard]}>
          <Text style={styles.actionIcon}>
            {band === 'HIGH' ? '🚫' : band === 'MEDIUM' ? '⚠️' : '✅'}
          </Text>
          <View style={{ flex: 1 }}>
            <Text style={styles.label}>Recommended action</Text>
            <Text style={[styles.actionValue, { color: bandColor(band) }]}>{action}</Text>
          </View>
        </View>

        {/* metrics row */}
        <View style={styles.metricsRow}>
          <View style={styles.metric}>
            <Text style={styles.label}>CNN PROB</Text>
            <Text style={styles.metricValue}>{cnn}</Text>
          </View>
          <View style={styles.metric}>
            <Text style={styles.label}>RISK SCORE</Text>
            <Text style={styles.metricValue}>{score}</Text>
          </View>
        </View>

        {/* analysis */}
        <View style={styles.card}>
          <Text style={styles.label}>Analysis</Text>
          <Text style={styles.body}>{explanation}</Text>
        </View>

        {/* flagged phrases — only shown when keywords hit */}
        {keywords.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.label}>Flagged phrases</Text>
            <View style={styles.chips}>
              {keywords.map((k) => (
                <View key={k} style={styles.chip}>
                  <Text style={styles.chipText}>{k}</Text>
                </View>
              ))}
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: 16, paddingBottom: 40, alignItems: 'center', gap: 10 },

  header: {
    flexDirection:     'row',
    alignItems:        'center',
    paddingHorizontal: 16,
    paddingVertical:   12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    gap: 8,
  },
  logoIcon:  {
    width: 30, height: 30,
    backgroundColor: colors.accent,
    borderRadius: 8,
    alignItems: 'center', justifyContent: 'center',
  },
  logoEmoji: { fontSize: 15 },
  logoText:  { flex: 1, fontSize: 17, fontWeight: '700', color: colors.text, letterSpacing: -0.5 },

  statusPill:     { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 20,
                    backgroundColor: colors.surface2, borderWidth: 1, borderColor: colors.border },
  statusPillLive: { borderColor: colors.high + '60', backgroundColor: colors.high + '15' },
  statusPillMuted:{ borderColor: colors.muted + '40' },
  statusText:     { fontSize: 10, fontWeight: '700', letterSpacing: 1, color: colors.muted },
  statusTextLive: { color: colors.high },

  gaugeWrap:     { marginVertical: 4 },

  micBtn: {
    width: 60, height: 60, borderRadius: 30,
    backgroundColor: colors.accent,
    alignItems: 'center', justifyContent: 'center',
  },
  micBtnDisabled: { backgroundColor: colors.surface2 },
  micEmoji:       { fontSize: 24 },

  noMicHint: {
    fontSize: 11,
    color: colors.muted,
    textAlign: 'center',
    marginTop: -4,
  },

  card: {
    width: '100%',
    backgroundColor: colors.surface,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: colors.border,
    gap: 5,
  },
  actionCard:  { flexDirection: 'row', alignItems: 'center', gap: 12 },
  actionIcon:  { fontSize: 26 },

  label:       { fontSize: 10, color: colors.muted, textTransform: 'uppercase', letterSpacing: 0.8 },
  actionValue: { fontSize: 14, fontWeight: '600', marginTop: 2 },
  body:        { fontSize: 13, color: '#94a3b8', lineHeight: 20 },

  metricsRow: { flexDirection: 'row', width: '100%', gap: 10 },
  metric: {
    flex: 1, backgroundColor: colors.surface2,
    borderRadius: 10, padding: 12, gap: 4,
  },
  metricValue: { fontSize: 19, fontWeight: '700', color: colors.text },

  chips:    { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 4 },
  chip:     { backgroundColor: 'rgba(239,68,68,0.15)', borderRadius: 20,
              paddingHorizontal: 10, paddingVertical: 3,
              borderWidth: 1, borderColor: 'rgba(239,68,68,0.25)' },
  chipText: { fontSize: 11, color: '#fca5a5' },
});
