import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { colors } from '../constants/theme';

export interface SessionSummary {
  id:          string;
  started_at:  string;
  ended_at:    string | null;
  event_count: number;
}

interface Props {
  session:  SessionSummary;
  onPress:  () => void;
}

export default function SessionCard({ session, onPress }: Props) {
  const start    = new Date(session.started_at);
  const duration = session.ended_at
    ? Math.round((new Date(session.ended_at).getTime() - start.getTime()) / 1000)
    : null;

  return (
    <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.row}>
        <Text style={styles.date}>
          {start.toLocaleDateString()} · {start.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </Text>
        {session.ended_at == null && (
          <View style={styles.liveDot} />
        )}
      </View>
      <View style={styles.row}>
        <Text style={styles.meta}>
          {session.event_count} events
          {duration != null ? `  ·  ${duration}s` : '  ·  ongoing'}
        </Text>
        <Text style={styles.arrow}>›</Text>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: colors.border,
    gap: 6,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  date: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '600',
  },
  meta: {
    color: colors.muted,
    fontSize: 13,
  },
  arrow: {
    color: colors.muted,
    fontSize: 20,
  },
  liveDot: {
    width: 8, height: 8,
    borderRadius: 4,
    backgroundColor: colors.high,
  },
});
