import React from 'react';
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, SafeAreaView, Animated,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import RingGauge from '../components/RingGauge';
import { useVoiceStream } from '../hooks/useVoiceStream';
import { colors, bandColor } from '../constants/theme';

export default function MonitorScreen() {
  const { status, event, start, stop } = useVoiceStream();

  const recording  = status === 'recording';
  const band       = event?.band  ?? 'LOW';
  const score      = event?.score ?? 0;
  const action     = event?.action ?? '—';
  const cnn        = event ? (event.cnn_prob * 100).toFixed(1) + '%' : '—';
  const transcript = event?.transcript || '—';
  const keywords   = event?.keywords ?? [];
  const explanation= event?.explanation || 'Start monitoring to analyse voice…';

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />

      {/* header */}
      <View style={styles.header}>
        <View style={styles.logoIcon}><Text style={styles.logoEmoji}>🛡️</Text></View>
        <Text style={styles.logoText}>VoiceGuard</Text>
        <View style={[styles.statusPill, recording && styles.statusPillLive]}>
          <Text style={[styles.statusText, recording && styles.statusTextLive]}>
            {recording ? '● LIVE' : status.toUpperCase()}
          </Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>

        {/* ring gauge */}
        <View style={styles.gaugeWrap}>
          <RingGauge score={score} band={band} />
        </View>

        {/* mic button */}
        <TouchableOpacity
          style={[styles.micBtn, recording && { backgroundColor: colors.high }]}
          onPress={recording ? stop : start}
          activeOpacity={0.8}
        >
          <Text style={styles.micEmoji}>{recording ? '⏹' : '🎙️'}</Text>
        </TouchableOpacity>

        {/* action card */}
        <View style={[styles.card, styles.actionCard]}>
          <Text style={styles.actionIcon}>
            {band === 'HIGH' ? '🚫' : band === 'MEDIUM' ? '⚠️' : '✅'}
          </Text>
          <View>
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

        {/* transcript */}
        <View style={styles.card}>
          <Text style={styles.label}>Live transcript</Text>
          <Text style={styles.body}>{transcript}</Text>
          {keywords.length > 0 && (
            <>
              <Text style={[styles.label, { marginTop: 12 }]}>Flagged phrases</Text>
              <View style={styles.chips}>
                {keywords.map((k) => (
                  <View key={k} style={styles.chip}>
                    <Text style={styles.chipText}>{k}</Text>
                  </View>
                ))}
              </View>
            </>
          )}
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: 16, paddingBottom: 32, alignItems: 'center', gap: 12 },

  header: {
    flexDirection:  'row',
    alignItems:     'center',
    padding:        16,
    paddingBottom:  12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    gap: 8,
  },
  logoIcon:  { width: 32, height: 32, backgroundColor: colors.accent, borderRadius: 8,
               alignItems: 'center', justifyContent: 'center' },
  logoEmoji: { fontSize: 16 },
  logoText:  { flex: 1, fontSize: 18, fontWeight: '700', color: colors.text, letterSpacing: -0.5 },
  statusPill: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 20,
                backgroundColor: colors.surface2, borderWidth: 1, borderColor: colors.border },
  statusPillLive: { borderColor: colors.high + '60', backgroundColor: colors.high + '15' },
  statusText:     { fontSize: 10, fontWeight: '700', letterSpacing: 1, color: colors.muted },
  statusTextLive: { color: colors.high },

  gaugeWrap: { marginVertical: 8 },

  micBtn: {
    width: 64, height: 64, borderRadius: 32,
    backgroundColor: colors.accent,
    alignItems: 'center', justifyContent: 'center',
    marginVertical: 4,
  },
  micEmoji: { fontSize: 26 },

  card: {
    width: '100%',
    backgroundColor: colors.surface,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: colors.border,
    gap: 6,
  },
  actionCard: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  actionIcon: { fontSize: 28 },

  label:        { fontSize: 10, color: colors.muted, textTransform: 'uppercase', letterSpacing: 0.8 },
  actionValue:  { fontSize: 15, fontWeight: '600', marginTop: 2 },
  body:         { fontSize: 13, color: '#94a3b8', lineHeight: 20 },

  metricsRow: { flexDirection: 'row', width: '100%', gap: 10 },
  metric: {
    flex: 1, backgroundColor: colors.surface2,
    borderRadius: 10, padding: 12, gap: 4,
  },
  metricValue: { fontSize: 20, fontWeight: '700', color: colors.text },

  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 4 },
  chip:  { backgroundColor: 'rgba(239,68,68,0.15)', borderRadius: 20,
           paddingHorizontal: 10, paddingVertical: 3,
           borderWidth: 1, borderColor: 'rgba(239,68,68,0.25)' },
  chipText: { fontSize: 11, color: '#fca5a5' },
});
