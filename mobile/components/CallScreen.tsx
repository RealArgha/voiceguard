/**
 * CallScreen — what the customer sees.
 *
 *   ┌──────────────────────────┐
 *   │   🛡️ VoiceGuard          │
 *   │                          │
 *   │      [ score 0-100 ]     │
 *   │       █████░░░░░░        │  ← gauge
 *   │                          │
 *   │   "Pass — call safe"     │
 *   │                          │
 *   │   [ 📞 Call my bank ]    │
 *   │                          │
 *   │   Explanation text…      │
 *   └──────────────────────────┘
 */

import { useEffect, useState } from 'react';
import {
  View, Text, Pressable, StyleSheet, ScrollView, Linking,
} from 'react-native';
import { useRiskFeed, RiskEvent } from '../lib/voiceguardApi';
import RiskGauge from './RiskGauge';

// ↓ Edit these two for your environment
const BANK_PHONE_NUMBER = '+15551234567';                  // your Twilio #
const BACKEND_WS_URL    = 'ws://10.0.0.10:8000/ws/dashboard'; // your laptop's LAN IP

const BAND_COLOR = { LOW: '#22c55e', MEDIUM: '#eab308', HIGH: '#ef4444' };
const ACTION_TEXT = {
  PASS:          '✓ Call appears safe',
  OTP_CHALLENGE: '⚠ Bank will request OTP verification',
  BLOCK:         '🛑 Call flagged — bank may block',
};

export default function CallScreen() {
  const risk: RiskEvent | null = useRiskFeed(BACKEND_WS_URL);
  const [callPlaced, setCallPlaced] = useState(false);

  const dial = () => {
    setCallPlaced(true);
    Linking.openURL(`tel:${BANK_PHONE_NUMBER}`);
  };

  const band  = risk?.band ?? 'LOW';
  const color = BAND_COLOR[band];

  return (
    <ScrollView contentContainerStyle={styles.scroll}
                style={{ backgroundColor: '#0f172a' }}>
      <Text style={styles.title}>🛡️ VoiceGuard</Text>
      <Text style={styles.sub}>Live AI-voice fraud monitor</Text>

      <View style={styles.card}>
        <Text style={styles.label}>Current call risk</Text>
        <RiskGauge score={risk?.score ?? 0} color={color} />
        <Text style={[styles.action, { color }]}>
          {risk ? ACTION_TEXT[risk.action] : 'Tap below to call'}
        </Text>
      </View>

      <Pressable style={styles.dialBtn} onPress={dial}>
        <Text style={styles.dialBtnText}>
          {callPlaced ? '📞 Call active' : '📞 Call my bank'}
        </Text>
      </Pressable>

      <View style={styles.card}>
        <Text style={styles.label}>Why</Text>
        <Text style={styles.body}>
          {risk?.explanation ?? 'No active call yet.'}
        </Text>
        {risk?.keywords?.length ? (
          <>
            <Text style={[styles.label, { marginTop: 12 }]}>
              Suspicious phrases
            </Text>
            <Text style={styles.body}>{risk.keywords.join(', ')}</Text>
          </>
        ) : null}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll:    { padding: 20, paddingTop: 60 },
  title:     { color: '#f1f5f9', fontSize: 28, fontWeight: '800' },
  sub:       { color: '#94a3b8', fontSize: 14, marginBottom: 24 },
  card:      { backgroundColor: '#1e293b', borderRadius: 12,
               padding: 20, marginBottom: 16,
               borderWidth: 1, borderColor: '#334155' },
  label:     { color: '#94a3b8', fontSize: 11,
               textTransform: 'uppercase', letterSpacing: 1,
               marginBottom: 8 },
  body:      { color: '#cbd5e1', fontSize: 15, lineHeight: 22 },
  action:    { fontSize: 16, fontWeight: '600', marginTop: 16,
               textAlign: 'center' },
  dialBtn:   { backgroundColor: '#3b82f6', borderRadius: 12,
               padding: 18, alignItems: 'center', marginBottom: 16 },
  dialBtnText:{ color: '#fff', fontSize: 17, fontWeight: '600' },
});
