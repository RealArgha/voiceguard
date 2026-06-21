import React, { useEffect } from 'react';
import {
  View, Text, FlatList, StyleSheet,
  SafeAreaView, ActivityIndicator, TouchableOpacity,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useSessionEvents } from '../../hooks/useSessions';
import { colors, bandColor, Band } from '../../constants/theme';

export default function SessionDetailScreen() {
  const { id }   = useLocalSearchParams<{ id: string }>();
  const router   = useRouter();
  const { events, loading, error, fetch } = useSessionEvents(id);

  useEffect(() => { fetch(); }, [id]);

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.back}>
          <Text style={styles.backText}>‹ Back</Text>
        </TouchableOpacity>
        <Text style={styles.title} numberOfLines={1}>Session</Text>
        <Text style={styles.sessionId} numberOfLines={1}>{id.slice(0, 8)}…</Text>
      </View>

      {loading && (
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} size="large" />
        </View>
      )}

      {error && (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity onPress={fetch} style={styles.retryBtn}>
            <Text style={styles.retryText}>Retry</Text>
          </TouchableOpacity>
        </View>
      )}

      {!loading && !error && events.length === 0 && (
        <View style={styles.center}>
          <Text style={styles.emptyText}>No events recorded.</Text>
        </View>
      )}

      {!loading && !error && events.length > 0 && (
        <FlatList
          data={events}
          keyExtractor={(e) => String(e.id)}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
          renderItem={({ item }) => {
            const band  = item.band as Band;
            const color = bandColor(band);
            const time  = new Date(item.created_at).toLocaleTimeString([], {
              hour: '2-digit', minute: '2-digit', second: '2-digit',
            });
            return (
              <View style={[styles.eventCard, { borderLeftColor: color }]}>
                <View style={styles.eventHeader}>
                  <Text style={[styles.eventBand, { color }]}>{band}</Text>
                  <Text style={styles.eventScore}>{item.score}/100</Text>
                  <Text style={styles.eventTime}>{time}</Text>
                </View>
                {item.transcript ? (
                  <Text style={styles.eventTranscript} numberOfLines={2}>
                    {item.transcript}
                  </Text>
                ) : null}
                {item.keywords?.length ? (
                  <View style={styles.chips}>
                    {item.keywords.map((k) => (
                      <View key={k} style={styles.chip}>
                        <Text style={styles.chipText}>{k}</Text>
                      </View>
                    ))}
                  </View>
                ) : null}
              </View>
            );
          }}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: colors.bg },

  header: {
    flexDirection:  'row',
    alignItems:     'center',
    padding:        16,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    gap: 8,
  },
  back:      { paddingRight: 8 },
  backText:  { color: colors.accent, fontSize: 16, fontWeight: '600' },
  title:     { flex: 1, fontSize: 18, fontWeight: '700', color: colors.text },
  sessionId: { fontSize: 11, color: colors.muted },

  list:   { padding: 16, paddingBottom: 32 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12 },

  errorText: { color: colors.high, fontSize: 14, textAlign: 'center', paddingHorizontal: 24 },
  retryBtn:  { backgroundColor: colors.accent, paddingHorizontal: 20, paddingVertical: 10,
               borderRadius: 8 },
  retryText: { color: '#fff', fontWeight: '600' },
  emptyText: { color: colors.muted, fontSize: 14 },

  eventCard: {
    backgroundColor: colors.surface,
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
    borderLeftWidth: 3,
    borderWidth: 1,
    borderColor: colors.border,
    gap: 6,
  },
  eventHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  eventBand:   { fontSize: 11, fontWeight: '700', letterSpacing: 1 },
  eventScore:  { flex: 1, fontSize: 13, fontWeight: '600', color: colors.text },
  eventTime:   { fontSize: 11, color: colors.muted },
  eventTranscript: { fontSize: 13, color: '#94a3b8', lineHeight: 18 },

  chips:    { flexDirection: 'row', flexWrap: 'wrap', gap: 4 },
  chip:     { backgroundColor: 'rgba(239,68,68,0.15)', borderRadius: 20,
              paddingHorizontal: 8, paddingVertical: 2 },
  chipText: { fontSize: 10, color: '#fca5a5' },
});
