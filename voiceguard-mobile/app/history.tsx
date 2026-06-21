import React, { useEffect } from 'react';
import {
  View, Text, FlatList, StyleSheet,
  SafeAreaView, ActivityIndicator, TouchableOpacity,
} from 'react-native';
import { useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import SessionCard from '../components/SessionCard';
import { useSessions } from '../hooks/useSessions';
import { colors } from '../constants/theme';

export default function HistoryScreen() {
  const { sessions, loading, error, fetch } = useSessions();
  const router = useRouter();

  useEffect(() => { fetch(); }, []);

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />

      <View style={styles.header}>
        <Text style={styles.title}>Session History</Text>
        <TouchableOpacity onPress={fetch} style={styles.refresh}>
          <Text style={styles.refreshText}>↻</Text>
        </TouchableOpacity>
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

      {!loading && !error && sessions.length === 0 && (
        <View style={styles.center}>
          <Text style={styles.emptyText}>No sessions yet.</Text>
          <Text style={styles.emptyHint}>Start monitoring to see history here.</Text>
        </View>
      )}

      {!loading && !error && sessions.length > 0 && (
        <FlatList
          data={sessions}
          keyExtractor={(s) => s.id}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
          renderItem={({ item }) => (
            <SessionCard
              session={item}
              onPress={() => router.push(`/session/${item.id}`)}
            />
          )}
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
    justifyContent: 'space-between',
    padding:        16,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  title:       { fontSize: 20, fontWeight: '700', color: colors.text },
  refresh:     { padding: 6 },
  refreshText: { fontSize: 22, color: colors.accent },

  list:      { padding: 16, paddingBottom: 32 },
  center:    { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12 },
  errorText: { color: colors.high, fontSize: 14, textAlign: 'center', paddingHorizontal: 24 },
  retryBtn:  { backgroundColor: colors.accent, paddingHorizontal: 20, paddingVertical: 10,
               borderRadius: 8 },
  retryText: { color: '#fff', fontWeight: '600' },

  emptyText: { color: colors.text,  fontSize: 16, fontWeight: '600' },
  emptyHint: { color: colors.muted, fontSize: 13 },
});
